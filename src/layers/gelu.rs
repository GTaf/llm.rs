use ndarray::Array2;

use crate::layers::{Layer, traits::Shape};
use async_trait::async_trait;

pub fn gelu(x: &f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / 2.0_f32.sqrt()))
}

pub struct Gelu {}

impl Gelu {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

#[async_trait]
impl Layer for Gelu {
    fn run_cpu(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        let result = input.map(gelu);
        Ok(result)
    }

    async fn run_gpu(&self, _: wgpu::Buffer, _: &Shape) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        todo!()
    }
}
