use std::sync::Arc;

use ndarray::{Array2, Array1};
use pollster::FutureExt;

use crate::gpu_backend::backend::{ComputePipeline, GpuBackend};


#[test]
fn test_gpu_backend_2x2() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec((2, 2), vec![1_f32, 2_f32, 3_f32, 4_f32])?;
    let bias = Array1::from_shape_vec(2, vec![1_f32, 2_f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec((2, 2), vec![1_f32, 2_f32, 3_f32, 4_f32])?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec((2, 2), vec![8_f32, 12_f32, 16_f32, 24_f32])?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_3x3_1() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 1_f32, 2_f32,
        ],
    )?;
    let bias = Array1::from_shape_vec(3, vec![0_f32, 0_f32, 0f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32,
        ],
    )?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec(
            (3, 3),
            vec![
                4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32
            ]
        )?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_3x3_2() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 2_f32, 1_f32,
        ],
    )?;
    let bias = Array1::from_shape_vec(3, vec![0_f32, 0_f32, 0f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 0_f32, 0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0_f32, 1_f32,
        ],
    )?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec(
            (3, 3),
            vec![
                1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 2_f32, 1_f32,
            ]
        )?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_3x3_3() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 0_f32, 0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0_f32, 1_f32,
        ],
    )?;
    let bias = Array1::from_shape_vec(3, vec![0_f32, 0_f32, 0f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec(
        (3, 3),
        vec![
            4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32,
        ],
    )?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec(
            (3, 3),
            vec![
                4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32
            ]
        )?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_4x4_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (4, 4),
        (0..16).map(|_| rng.gen_range(0., 4.)).collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(4, vec![0_f32, 0_f32, 0f32, 0_f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (4, 4),
        (0..16).map(|_| rng.gen_range(0., 4.)).collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_32x32_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let matrix_size = 32;
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (matrix_size, matrix_size),
        (0..matrix_size * matrix_size)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(matrix_size, vec![0_f32; matrix_size])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (matrix_size, matrix_size),
        (0..matrix_size * matrix_size)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_3_2_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (3, 2, 1);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_32_10_11_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (32, 10, 11);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_32_32_32_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (32, 32, 32);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_16_16_16() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (16, 16, 16);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_33() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (33, 33, 33);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}
