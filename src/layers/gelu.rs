use std::sync::Arc;

use ndarray::Array2;

use crate::{
    gpu_backend::{backend::GpuBackend, pipelines::gelu_pipeline::GeLUComputePipeline},
    layers::{Layer, traits::Shape},
};
use async_trait::async_trait;

pub fn gelu(x: &f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / 2.0_f32.sqrt()))
}

pub struct Gelu {
    pipeline: Option<GeLUComputePipeline>,
}

impl Gelu {
    pub fn new(gpu_backend: Option<Arc<GpuBackend>>) -> anyhow::Result<Self> {
        Ok(match gpu_backend {
            Some(backend) => {
                let pipeline = Some(GeLUComputePipeline::new_pipeline(backend));
                Self { pipeline }
            }
            None => Self { pipeline: None },
        })
    }
}

#[async_trait]
impl Layer for Gelu {
    fn run_cpu(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        let result = input.map(gelu);
        Ok(result)
    }

    async fn run_gpu(
        &self,
        buffer: &wgpu::Buffer,
        shape: &Shape,
    ) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        match &self.pipeline {
            Some(pipeline) => pipeline.compute(buffer, shape).await,
            None => panic!("Souldn't use GPU data with CPU layer"),
        }
    }
}
