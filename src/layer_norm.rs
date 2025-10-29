use ndarray::{Array1, Array2, Axis};
use safetensors::tensor::TensorView;

use crate::tools::weights_to_array1;

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

    pub fn run(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros((input.shape()[0], input.shape()[1]));
        for (i, input_row) in input.axis_iter(Axis(0)).enumerate() {
            let n = input_row.len() as f32;

            // Calculer mean
            let mean = input_row.mean().unwrap();

            // Calculer variance (diviseur n, pas n-1)
            let var = input_row.mapv(|x| (x - mean).powi(2)).sum() / n;

            // Normalisation : (x - mean) / sqrt(var + eps)
            let normalized = input_row.mapv(|x| (x - mean) / (var + 1e-5).sqrt());

            // Appliquer weight et bias (element-wise)
            result
                .row_mut(i)
                .assign(&(&normalized * &self.weight + &self.bias));
        }

        result
    }
}
