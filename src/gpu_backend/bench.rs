use llmrs::gpu_backend::backend::{ComputePipeline, GpuBackend};
use ndarray::{Array1, Array2};
use pollster::FutureExt;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    use rand::Rng;
    for _ in 0..10 {
        let mut rng = rand::thread_rng();
        let (m, k, n) = (2048, 1024, 128);
        let backend = Arc::new(GpuBackend::new().block_on()?);

        let weights = Array2::from_shape_vec(
            (k, n),
            (0..k * n)
                .map(|_| rng.gen_range(0., 4.))
                .collect::<Vec<f32>>(),
        )?;

        let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
        let compute_pipeline =
            ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
        let input = Array2::from_shape_vec(
            (m, k),
            (0..m * k)
                .map(|_| rng.gen_range(0., 4.))
                .collect::<Vec<f32>>(),
        )?;
        let _output = compute_pipeline.compute(input.clone()).block_on()?;
    }
    // assert_eq!(output, input.dot(&weights));
    Ok(())
}
