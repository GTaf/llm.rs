use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

use crate::tools::weights_to_array1;

pub struct LayerNorm {
    bias: Array1<f32>,
    weight: Array1<f32>,
}

impl LayerNorm {
    pub fn new(weights: TensorView, bias: TensorView) -> Self {
        Self {
            bias: weights_to_array1(&bias),
            weight: weights_to_array1(&weights),
        }
    }

    pub fn run(self, input: &Array2<f32>) -> Array2<f32> {
        todo!();
    }
}
