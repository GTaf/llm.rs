use std::sync::Arc;

use crate::gpu_backend::{ComputePipeline, GpuBackend};
use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

use crate::tools::{weights_to_array, weights_to_array1};

pub enum LinearLayer {
    Cpu(CpuLinearLayer),
    Gpu(GpuLinearLayer),
}

impl LinearLayer {
    pub async fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        match self {
            LinearLayer::Cpu(cpu) => cpu.run(input),
            LinearLayer::Gpu(gpu) => gpu.run(input).await,
        }
    }
}

pub struct GpuLinearLayer {
    compute_pipeline: ComputePipeline,
}

pub struct CpuLinearLayer {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl CpuLinearLayer {
    pub fn new(weights: TensorView, bias: TensorView) -> anyhow::Result<Self> {
        Ok(Self {
            weight: weights_to_array(&weights)?,
            bias: weights_to_array1(&bias)?,
        })
    }

    pub fn new_no_bias(weights: TensorView) -> anyhow::Result<Self> {
        let weight = weights_to_array(&weights)?.t().to_owned();
        let bias = Array1::zeros(weight.shape()[1]);
        Ok(Self { weight, bias })
    }
}

impl GpuLinearLayer {
    pub fn new(
        backend: Arc<GpuBackend>,
        weights: TensorView,
        bias: TensorView,
    ) -> anyhow::Result<Self> {
        let compute_pipeline = ComputePipeline::new_pipeline(
            backend.clone(),
            weights_to_array(&weights)?,
            weights_to_array1(&bias)?,
        );
        Ok(Self { compute_pipeline })
    }

    pub fn _new_no_bias(backend: Arc<GpuBackend>, weights: TensorView) -> anyhow::Result<Self> {
        let weight = weights_to_array(&weights)?.t().to_owned();
        let bias = Array1::zeros(weight.shape()[1]);
        let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weight, bias);
        Ok(Self { compute_pipeline })
    }
}

impl GpuLinearLayer {
    async fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        self.compute_pipeline.compute(input.to_owned()).await
    }
}

impl CpuLinearLayer {
    fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        Ok(input.dot(&self.weight) + &self.bias)
    }
}
