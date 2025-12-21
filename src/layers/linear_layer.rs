use std::sync::Arc;

use crate::{
    gpu_backend::{backend::GpuBackend, pipelines::linear_pipeline::LinearComputePipeline},
    layers::{Layer, traits::Shape},
};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

use crate::tools::{weights_to_array, weights_to_array1};

pub enum LinearLayer {
    Cpu(CpuLinearLayer),
    Gpu(GpuLinearLayer),
}

pub struct GpuLinearLayer {
    compute_pipeline: LinearComputePipeline,
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
        let compute_pipeline = LinearComputePipeline::new_pipeline(
            backend.clone(),
            weights_to_array(&weights)?,
            weights_to_array1(&bias)?,
        );
        Ok(Self { compute_pipeline })
    }

    pub fn _new_no_bias(backend: Arc<GpuBackend>, weights: TensorView) -> anyhow::Result<Self> {
        let weight = weights_to_array(&weights)?.t().to_owned();
        let bias = Array1::zeros(weight.shape()[1]);
        let compute_pipeline = LinearComputePipeline::new_pipeline(backend.clone(), weight, bias);
        Ok(Self { compute_pipeline })
    }
}

impl GpuLinearLayer {
    async fn run(
        &self,
        input: &wgpu::Buffer,
        shape: &Shape,
    ) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        self.compute_pipeline.compute(input, shape).await
    }
}

impl CpuLinearLayer {
    fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        Ok(input.dot(&self.weight) + &self.bias)
    }
}

#[async_trait]
impl Layer for LinearLayer {
    fn run_cpu(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        if let LinearLayer::Cpu(layer) = self {
            layer.run(input)
        } else {
            todo!()
        }
    }

    async fn run_gpu(
        &self,
        input: &wgpu::Buffer,
        shape: &Shape,
    ) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        if let LinearLayer::Gpu(layer) = self {
            layer.run(input, shape).await
        } else {
            todo!()
        }
    }
}
