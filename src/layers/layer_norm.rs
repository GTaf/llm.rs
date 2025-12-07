use std::sync::Arc;

use ndarray::{Array1, Array2, Axis};
use safetensors::tensor::TensorView;

use crate::{
    gpu_backend::{backend::GpuBackend, pipelines::layer_norm_pipeline::LayerNormComputePipeline},
    layers::{Layer, traits::Shape},
    tools::weights_to_array1,
};
use async_trait::async_trait;

pub struct LayerNorm {
    pipeline: Option<LayerNormComputePipeline>,
    bias: Array1<f32>,
    weight: Array1<f32>,
}

impl LayerNorm {
    pub fn new(
        weights: TensorView,
        bias: TensorView,
        gpu_backend: Option<Arc<GpuBackend>>,
    ) -> anyhow::Result<Self> {
        let bias = weights_to_array1(&bias)?;
        let weight = weights_to_array1(&weights)?;
        let pipeline = match gpu_backend {
            Some(backend) => Some(LayerNormComputePipeline::new_pipeline(
                backend,
                weight.clone(),
                bias.clone(),
            )),
            None => None,
        };
        Ok(Self {
            pipeline,
            bias,
            weight,
        })
    }
}

#[async_trait]
impl Layer for LayerNorm {
    fn run_cpu(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        let mut result = Array2::zeros((input.shape()[0], input.shape()[1]));
        for (i, input_row) in input.axis_iter(Axis(0)).enumerate() {
            let n = input_row.len() as f32;

            let mean = input_row.mean().unwrap();

            let var = input_row.mapv(|x| (x - mean).powi(2)).sum() / n;

            let normalized = input_row.mapv(|x| (x - mean) / (var + 1e-5).sqrt());

            result
                .row_mut(i)
                .assign(&(&normalized * &self.weight + &self.bias));
        }

        Ok(result)
    }

    async fn run_gpu(
        &self,
        buffer: wgpu::Buffer,
        shape: &Shape,
    ) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        match &self.pipeline {
            Some(pipeline) => pipeline.compute(&buffer, shape).await,
            None => panic!("Souldn't use GPU data with CPU layer"),
        }
    }
}
