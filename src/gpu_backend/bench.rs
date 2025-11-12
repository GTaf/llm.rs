use llmrs::gpu_backend::backend::{ComputePipeline, GpuBackend};
use ndarray::{Array1, Array2};
use pollster::FutureExt;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    use rand::Rng;
    let mut timestamps = vec![0_f64; 10];
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
    for i in 0..10 {
        let compute_pipeline =
            ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias.clone());
        let _output = compute_pipeline
            .compute_timestamp(input.clone(), Some(&mut timestamps[i]))
            .block_on()?;
    }

    let ops = (2 * m * n * k) as f64;
    let time = timestamps.iter().sum::<f64>() as f64 / timestamps.len() as f64;
    let gflops = ops / time;
    println!("Results :\nTime : {time}µs and Ops : {ops}\nGFLOPs : {gflops}");
    // assert_eq!(output, input.dot(&weights));
    Ok(())
}
