use ndarray::{Array1, Array2, Axis};
use safetensors::tensor::TensorView;

use crate::{layers::Layer, tools::weights_to_array1};

pub struct RMSNorm {
    weight: Array1<f32>,
}

impl RMSNorm {
    pub fn new(weights: TensorView) -> anyhow::Result<Self> {
        Ok(Self {
            weight: weights_to_array1(&weights)?,
        })
    }
}

impl Layer for RMSNorm {
    fn run(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros((input.shape()[0], input.shape()[1]));
        for (i, input_row) in input.axis_iter(Axis(0)).enumerate() {
            let n = input_row.len() as f32;
            let var = input_row.mapv(|x| x.powi(2)).sum() / n;
            let rms = (var + 1e-5).sqrt();
            let normalized = input_row.mapv(|x| x/rms);
            result
                .row_mut(i)
                .assign(&(&normalized * &self.weight));
        }

        result
    }
}
