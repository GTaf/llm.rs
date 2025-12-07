use std::sync::Arc;

use ndarray::{Array1, Array2};
use pollster::FutureExt;
use wgpu::{
    Buffer, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    gpu_backend::{
        ComputeShape, backend::GpuBackend, pipelines::linear_pipeline::LinearComputePipeline,
    },
    layers::traits::Shape,
};

fn ndarray_to_gpubuffer(input: &Array2<f32>, device: &Device) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input_buffer"),
        contents: bytemuck::cast_slice(input.as_slice().unwrap()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    })
}

fn gpubuffer_to_ndarray(input: Buffer, shape: Shape) -> Array2<f32> {
    let output_data = input.get_mapped_range(..);
    let raw_data: &[f32] = bytemuck::cast_slice(&output_data);

    Array2::from_shape_vec(
        (shape.rows as usize, shape.columns as usize),
        raw_data.to_vec(),
    )
    .unwrap()
}

// Main test function for matrix multiplication on GPU
fn test_gpu_matmul(
    shape: ComputeShape,
    weights: Vec<f32>,
    bias: Vec<f32>,
    input: Vec<f32>,
    expected: Option<Vec<f32>>,
) -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec((shape.k as usize, shape.n as usize), weights)?;
    let bias = Array1::from_shape_vec(shape.n as usize, bias)?;
    let compute_pipeline =
        LinearComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec((shape.m as usize, shape.k as usize), input)?;
    let output = compute_pipeline
        .compute(
            &ndarray_to_gpubuffer(&input, &backend.device),
            &Shape {
                columns: shape.k as usize,
                rows: shape.m as usize,
            },
        )
        .block_on()?;
    let result_array = gpubuffer_to_ndarray(output.0, output.1);
    if let Some(expected) = expected {
        let expected = Array2::from_shape_vec((shape.m as usize, shape.n as usize), expected)?;
        assert_eq!(result_array, expected);
    } else {
        assert_eq!(result_array, input.dot(&weights));
    }
    Ok(())
}

#[test]
fn test_gpu_2x2() -> anyhow::Result<()> {
    test_gpu_matmul(
        ComputeShape { m: 2, k: 2, n: 2 },
        vec![1_f32, 2_f32, 3_f32, 4_f32],
        vec![1_f32, 2_f32],
        vec![1_f32, 2_f32, 3_f32, 4_f32],
        Some(vec![8_f32, 12_f32, 16_f32, 24_f32]),
    )
}

#[test]
fn test_gpu_3x3_1() -> anyhow::Result<()> {
    test_gpu_matmul(
        ComputeShape { m: 3, k: 3, n: 3 },
        vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 1_f32, 2_f32,
        ],
        vec![0_f32, 0_f32, 0_f32],
        vec![
            1_f32, 0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32,
        ],
        Some(vec![
            4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32,
        ]),
    )
}

#[test]
fn test_gpu_3x3_2() -> anyhow::Result<()> {
    test_gpu_matmul(
        ComputeShape { m: 3, k: 3, n: 3 },
        vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 2_f32, 1_f32,
        ],
        vec![0_f32, 0_f32, 0_f32],
        vec![
            1_f32, 0_f32, 0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0_f32, 1_f32,
        ],
        Some(vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 2_f32, 1_f32,
        ]),
    )
}

#[test]
fn test_gpu_3x3_3() -> anyhow::Result<()> {
    test_gpu_matmul(
        ComputeShape { m: 3, k: 3, n: 3 },
        vec![
            1_f32, 0_f32, 0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0_f32, 1_f32,
        ],
        vec![0_f32, 0_f32, 0_f32],
        vec![
            4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32,
        ],
        Some(vec![
            4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32,
        ]),
    )
}

#[test]
fn test_gpu_4x4_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let shape = ComputeShape { m: 4, k: 4, n: 4 };
    let weights: Vec<f32> = (0..16).map(|_| rng.gen_range(0., 4.)).collect();
    let bias = vec![0_f32; 4];
    let input: Vec<f32> = (0..16).map(|_| rng.gen_range(0., 4.)).collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}

#[test]
fn test_gpu_32x32_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let matrix_size = 32;
    let shape = ComputeShape {
        m: matrix_size,
        k: matrix_size,
        n: matrix_size,
    };
    let weights: Vec<f32> = (0..matrix_size * matrix_size)
        .map(|_| rng.gen_range(0., 4.))
        .collect();
    let bias = vec![0_f32; matrix_size as usize];
    let input: Vec<f32> = (0..matrix_size * matrix_size)
        .map(|_| rng.gen_range(0., 4.))
        .collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}

#[test]
fn test_gpu_3x2x1_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (3, 2, 1);
    let shape = ComputeShape { m, k, n };
    let weights: Vec<f32> = (0..k * n).map(|_| rng.gen_range(0., 4.)).collect();
    let bias = vec![0_f32; n as usize];
    let input: Vec<f32> = (0..m * k).map(|_| rng.gen_range(0., 4.)).collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}

#[test]
fn test_gpu_32x10x11_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (32, 10, 11);
    let shape = ComputeShape { m, k, n };
    let weights: Vec<f32> = (0..k * n).map(|_| rng.gen_range(0., 4.)).collect();
    let bias = vec![0_f32; n as usize];
    let input: Vec<f32> = (0..m * k).map(|_| rng.gen_range(0., 4.)).collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}

#[test]
fn test_gpu_32x32x32_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (32, 32, 32);
    let shape = ComputeShape { m, k, n };
    let weights: Vec<f32> = (0..k * n).map(|_| rng.gen_range(0., 4.)).collect();
    let bias = vec![0_f32; n as usize];
    let input: Vec<f32> = (0..m * k).map(|_| rng.gen_range(0., 4.)).collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}

#[test]
fn test_gpu_16x16x16_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (16, 16, 16);
    let shape = ComputeShape { m, k, n };
    let weights: Vec<f32> = (0..k * n).map(|_| rng.gen_range(0., 4.)).collect();
    let bias = vec![0_f32; n as usize];
    let input: Vec<f32> = (0..m * k).map(|_| rng.gen_range(0., 4.)).collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}

#[test]
fn test_gpu_33x33x33_random() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (33, 33, 33);
    let shape = ComputeShape { m, k, n };
    let weights: Vec<f32> = (0..k * n).map(|_| rng.gen_range(0., 4.)).collect();
    let bias = vec![0_f32; n as usize];
    let input: Vec<f32> = (0..m * k).map(|_| rng.gen_range(0., 4.)).collect();
    test_gpu_matmul(shape, weights, bias, input, None)
}
