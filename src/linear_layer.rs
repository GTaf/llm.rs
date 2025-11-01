use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

use crate::tools::{weights_to_array, weights_to_array1};
pub struct LinearLayer {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl LinearLayer {
    pub fn new(weights: TensorView, bias: TensorView) -> anyhow::Result<Self> {
        Ok(Self {
            weight: weights_to_array(&weights)?,
            bias: weights_to_array1(&bias)?,
        })
    }

    pub fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        Ok(input.dot(&self.weight) + &self.bias)
    }
}
