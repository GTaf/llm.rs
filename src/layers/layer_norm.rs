use ndarray::{Array1, Array2, Axis};
use safetensors::tensor::TensorView;

use crate::{layers::Layer, tools::weights_to_array1};

pub struct LayerNorm {
    bias: Array1<f32>,
    weight: Array1<f32>,
}

impl LayerNorm {
    pub fn new(weights: TensorView, bias: TensorView) -> anyhow::Result<Self> {
        Ok(Self {
            bias: weights_to_array1(&bias)?,
            weight: weights_to_array1(&weights)?,
        })
    }
}

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
    
    fn run_gpu(&self, _: wgpu::Buffer) -> anyhow::Result<wgpu::Buffer> {
        todo!()
    }
}
