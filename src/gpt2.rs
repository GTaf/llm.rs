use ndarray::{Array1, Array2};
use safetensors::SafeTensors;

use crate::{
    attention_block::AttentionBlock, embedding_layer::EmbeddingLayer, layer_norm::LayerNorm,
    linear_layer::LinearLayer,
};

pub struct GPT2 {
    pub embedding_layer: EmbeddingLayer,
    pub attention_blocks: Vec<AttentionBlock>,
    pub layer_norm: LayerNorm,
    pub linear_layer: LinearLayer,
}

impl GPT2 {
    pub fn new(tensor_weights: &SafeTensors) -> anyhow::Result<Self> {
        let mut attention_blocks = Vec::new();
        for i in 0..12 {
            attention_blocks.push(AttentionBlock::new(tensor_weights, i)?);
        }
        Ok(Self {
            embedding_layer: EmbeddingLayer::new(tensor_weights)?,
            attention_blocks,
            layer_norm: LayerNorm::new(
                tensor_weights.tensor("ln_f.weight")?,
                tensor_weights.tensor("ln_f.bias")?,
            )?,
            linear_layer: LinearLayer::new_no_bias(tensor_weights.tensor("wte.weight")?)?,
        })
    }

    pub fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array1<f32>> {
        let mut res = input.clone();
        for i in 0..12 {
            res = self.attention_blocks[i].run(&res)?;
        }
        res = self.layer_norm.run(&res);
        let res = self
            .linear_layer
            .run(&res)?
            .row(res.shape()[0] - 1)
            .to_owned();
        Ok(res)
    }
}
