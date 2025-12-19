use std::sync::Arc;

use ndarray::Array2;
use wgpu::Buffer;

use crate::{gpu_backend::{backend::GpuBackend, pipelines::element_wise_add::ElementWiseAddPipeline}, layers::traits::{Shape, TensorData}};


pub struct Tensor {
    pub data: TensorData,
    pub shape: Shape,
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

    pub fn data_move(self) -> TensorData {
        self.data
    }

    pub fn data_gpu(&self) -> &Buffer {
        match &self.data {
            TensorData::CpuData(_) => todo!(),
            TensorData::GpuData(buffer) => buffer,
        }
    }

    pub fn data_gpu_move(self) -> Buffer {
        match self.data {
            TensorData::CpuData(_) => todo!(),
            TensorData::GpuData(buffer) => buffer,
        }
    }

    pub fn data_cpu(&self) -> &Array2<f32> {
        match &self.data {
            TensorData::CpuData(array_base) => array_base,
            TensorData::GpuData(_) => todo!(),
        }
    }

    pub fn data_cpu_move(self) -> Array2<f32> {
        match self.data {
            TensorData::CpuData(array_base) => array_base,
            TensorData::GpuData(_) => todo!(),
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub async fn add(tensor1: Self, tensor2: Self, backend: Option<Arc<GpuBackend>>) -> anyhow::Result<Self> {
        // Check that both tensors are on the same backend
        if std::mem::discriminant(&tensor1.data) != std::mem::discriminant(&tensor2.data) {
            panic!("Cannot add tensors on different backends (CPU vs GPU)");
        }
        
        // Check that shapes match
        if tensor1.shape != tensor2.shape {
            panic!(
                "Cannot add tensors with different shapes: {:?} vs {:?}",
                tensor1.shape, tensor2.shape
            );
        }
        
        Ok(match tensor1.data {
            TensorData::CpuData(array1) => {
                // Extract CPU data from tensor2
                let TensorData::CpuData(array2) = tensor2.data else {
                    unreachable!("Already checked discriminant matches")
                };
                
                // Perform CPU addition
                let result_array = array1 + array2;
                Tensor::new_cpu(result_array)
            }
            
            TensorData::GpuData(gpu_data1) => {
                // Extract GPU data from tensor2
                let shape = tensor2.shape().clone();
                let TensorData::GpuData(gpu_data2) = tensor2.data else {
                    unreachable!("Already checked discriminant matches")
                };

                let pipeline = ElementWiseAddPipeline::new_pipeline(backend.unwrap());
                let (buf, shape) = pipeline.compute(&gpu_data1, &gpu_data2, &shape).await?;
                Tensor::new_gpu(buf, shape)
            }
        })
    }
}
