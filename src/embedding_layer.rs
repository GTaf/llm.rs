use ndarray::Array2;
use safetensors::tensor::TensorView;

use crate::tools::weights_to_array;

pub struct EmbeddingLayer {
    dimension: usize,
    wpe_f32: Array2<f32>,
    wte_f32: Array2<f32>,
}

impl EmbeddingLayer {
    pub fn new(dimension: usize, wte: TensorView, wpe: TensorView) -> Self {
        let wpe_f32 = weights_to_array(&wpe);
        let wte_f32 = weights_to_array(&wte);

        EmbeddingLayer {
            dimension,
            wte_f32,
            wpe_f32,
        }
    }

    pub fn run(self, tokens: &[u32]) -> Array2<f32> {
        let mut result = Array2::zeros((tokens.len(), self.dimension));
        for p in 0..tokens.len() {
            result.row_mut(p).scaled_add(1., &self.wpe_f32.row(p));
            result
                .row_mut(p)
                .scaled_add(1., &self.wte_f32.row(tokens[p] as usize));
        }
        result
    }
}
