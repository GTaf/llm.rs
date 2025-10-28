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
        let mut result = Array2::zeros((input.shape()[0], input.shape()[1]));
        let iter = result.rows_mut().into_iter().zip(input.rows());
        for (mut res_vec, input_vec) in iter {
            let mean = input_vec.mean().unwrap();
            let var = input_vec.std(0_f32).powi(2);
            res_vec.assign(&input_vec);
            res_vec.scaled_add(-mean, &Array1::ones(input.shape()[1]));
            res_vec /= (var + 1e-05).sqrt();
        }

        result.dot(&self.weight);
        result.scaled_add(1_f32, &self.bias);
        result
    }
}
