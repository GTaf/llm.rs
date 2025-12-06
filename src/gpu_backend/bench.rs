use llmrs::gpu_backend::backend::{ComputePipeline, GpuBackend};
use llmrs::layers::traits::{Shape, Tensor};
use ndarray::{Array1, Array2};
use pollster::FutureExt;
use std::sync::Arc;
use wgpu::{
    Buffer, Device,
    util::{BufferInitDescriptor, DeviceExt},
};

fn ndarray_to_gpubuffer(input: &Array2<f32>, device: &Device) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input_buffer"),
        contents: bytemuck::cast_slice(input.as_slice().unwrap()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    })
}

fn main() -> anyhow::Result<()> {
    use rand::Rng;
    let mut timestamps = [0_f64; 10];
    let (m, k, n) = (2048, 2048, 2048);
    let mut rng = rand::thread_rng();
    let backend = Arc::new(GpuBackend::new().block_on()?);

    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let input_buffer = ndarray_to_gpubuffer(&input, &backend.device);
    for timestamp in timestamps.iter_mut().take(10) {
        let compute_pipeline =
            ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias.clone());
        let _output = compute_pipeline
            .compute_timestamp(
                &input_buffer,
                Some(timestamp),
                &Shape {
                    columns: m,
                    rows: k,
                },
            )
            .block_on()?;
    }

    let ops = (2 * m * n * k) as f64;
    let time = timestamps.iter().sum::<f64>() / timestamps.len() as f64;
    let gflops = ops / time;
    println!("Results :\nTime : {time}µs and Ops : {ops}\nGFLOPs : {gflops}");
    // assert_eq!(output, input.dot(&weights));
    Ok(())
}
