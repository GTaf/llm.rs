use ndarray::Array2;
use safetensors::SafeTensors;

use crate::tools::weights_to_array;

pub struct EmbeddingLayer {
    wpe_f32: Array2<f32>,
    wte_f32: Array2<f32>,
}

impl EmbeddingLayer {
    pub fn new(tensor_weights: &SafeTensors) -> anyhow::Result<Self> {
        let wte = tensor_weights.tensor("wte.weight")?;
        let wpe = tensor_weights.tensor("wpe.weight")?;
        let wpe_f32 = weights_to_array(&wpe)?;
        let wte_f32 = weights_to_array(&wte)?;

        Ok(EmbeddingLayer { wte_f32, wpe_f32 })
    }

    pub fn run(&self, tokens: &[u32]) -> Array2<f32> {
        let dimension = self.wpe_f32.shape()[1];
        let mut result = Array2::zeros((tokens.len(), dimension));
        for (p, token) in tokens.iter().enumerate() {
            result.row_mut(p).scaled_add(1., &self.wpe_f32.row(p));
            result
                .row_mut(p)
                .scaled_add(1., &self.wte_f32.row(*token as usize));
        }
        result
    }
}
