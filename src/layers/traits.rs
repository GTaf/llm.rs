use ndarray::Array2;
use wgpu::Buffer;

pub enum TensorData {
    CpuData(Array2<f32>),
    GpuData(Buffer),
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

pub trait Layer: Send + Sync {
    fn run_cpu(&self, input: &Array2<f32>) -> Array2<f32>;
    fn run_gpu(&self, input: Buffer) -> Buffer;
}

impl dyn Layer {
    pub fn run(&self, input: TensorData) -> TensorData {
        match input {
            TensorData::CpuData(array_base) => TensorData::CpuData(self.run_cpu(&array_base)),
            TensorData::GpuData(buffer) => TensorData::GpuData(self.run_gpu(buffer)),
        }
        
    }
}
