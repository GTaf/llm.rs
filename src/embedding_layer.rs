use ndarray::Array2;
use safetensors::SafeTensors;

use crate::tools::weights_to_array;

pub struct EmbeddingLayer {
    wpe_f32: Option<Array2<f32>>,
    wte_f32: Array2<f32>,
}

impl EmbeddingLayer {
    pub fn new(tensor_weights: &SafeTensors, position_enconding_name: Option<&str>, token_embedding: &str) -> anyhow::Result<Self> {
        let wte = tensor_weights.tensor(token_embedding)?;
        let wte_f32 = weights_to_array(&wte)?;
        let wpe_f32 = if let Some(name) = position_enconding_name {
            let wpe = tensor_weights.tensor(name)?;
            Some(weights_to_array(&wpe)?)} 
            else {None};

        Ok(EmbeddingLayer { wte_f32, wpe_f32 })
    }

    pub fn run(&self, tokens: &[u32]) -> Array2<f32> {
        let dimension = self.wte_f32.shape()[1];
        let mut result = Array2::zeros((tokens.len(), dimension));
        for (p, token) in tokens.iter().enumerate() {
            if self.wpe_f32.is_some() {
            result.row_mut(p).scaled_add(1., &self.wpe_f32.as_ref().unwrap().row(p));
            }
            result
                .row_mut(p)
                .scaled_add(1., &self.wte_f32.row(*token as usize));
        }
        result
    }
}
