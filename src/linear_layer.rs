use ndarray::Array2;
use safetensors::tensor::TensorView;

use crate::tools::weights_to_array;
pub struct LinearLayer {
    weight: Array2<f32>,
    bias: Array2<f32>,
}

impl LinearLayer {
    pub fn new(weights: TensorView, bias: TensorView) -> anyhow::Result<Self> {
        Ok(Self {
            weight: weights_to_array(&weights)?,
            bias: weights_to_array(&bias)?,
        })
    }

    pub fn run(self, input: Array2<f32>) -> Array2<f32> {
        input.dot(&self.weight) + self.bias
    }
}
