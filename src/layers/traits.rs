use async_trait::async_trait;
use ndarray::Array2;
use wgpu::Buffer;

pub enum TensorData {
    CpuData(Array2<f32>),
    GpuData(Buffer),
}

#[derive(Debug, Clone)]
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

pub struct Tensor {
    data: TensorData,
    shape: Shape,
}

impl Tensor {
    pub fn new_cpu(data: Array2<f32>) -> Self {
        let shape = Shape::from(&data);
        Self {
            data: TensorData::CpuData(data),
            shape,
        }
    }

    pub fn new_gpu(data: Buffer, shape: Shape) -> Self {
        Self {
            data: TensorData::GpuData(data),
            shape,
        }
    }

    pub fn data(&self) -> &TensorData {
        &self.data
    }

    pub fn data_gpu(&self) -> &Buffer {
        match &self.data {
            TensorData::CpuData(_) => todo!(),
            TensorData::GpuData(buffer) => &buffer,
        }
    }

    pub fn data_cpu(&self) -> &Array2<f32> {
        match &self.data {
            TensorData::CpuData(array_base) => &array_base,
            TensorData::GpuData(_) => todo!(),
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
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
                let result_shape = result.shape();
                let shape = Shape {
                    columns: result_shape[0],
                    rows: result_shape[1],
                };
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
