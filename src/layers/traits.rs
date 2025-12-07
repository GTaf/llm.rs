use async_trait::async_trait;
use ndarray::Array2;
use wgpu::Buffer;

use crate::gpu_backend::tensor::Tensor;

pub enum TensorData {
    CpuData(Array2<f32>),
    GpuData(Buffer),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub columns: usize,
    pub rows: usize,
}

impl<T> From<&Array2<T>> for Shape {
    fn from(value: &Array2<T>) -> Self {
        Shape {
            rows: value.shape()[0],
            columns: value.shape()[1],
        }
    }
}

impl From<Array2<f32>> for TensorData {
    fn from(arr: Array2<f32>) -> Self {
        TensorData::CpuData(arr)
    }
}

impl From<Buffer> for TensorData {
    fn from(buf: Buffer) -> Self {
        TensorData::GpuData(buf)
    }
}

#[async_trait]
pub trait Layer: Send + Sync {
    fn run_cpu(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>>;
    async fn run_gpu(&self, input: Buffer, shape: &Shape) -> anyhow::Result<(Buffer, Shape)>;
}

impl dyn Layer {
    pub async fn run(&self, input: Tensor) -> anyhow::Result<Tensor> {
        let shape = (*input.shape()).clone();
        let tensor = match input.data {
            TensorData::CpuData(array_base) => {
                let result = self.run_cpu(&array_base)?;
                let shape = Shape::from(&result);
                Tensor {
                    data: TensorData::CpuData(result),
                    shape,
                }
            }
            TensorData::GpuData(buffer) => {
                let (data, shape) = self.run_gpu(buffer, &shape).await?;
                Tensor {
                    data: TensorData::GpuData(data),
                    shape,
                }
            }
        };
        Ok(tensor)
    }
}
