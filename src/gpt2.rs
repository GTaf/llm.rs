use safetensors::SafeTensors;

use crate::{attention_block::AttentionBlock, embedding_layer::EmbeddingLayer};

pub struct GPT2 {
    pub embedding_layer: EmbeddingLayer,
    pub attention_blocks: Vec<AttentionBlock>,
}

impl GPT2 {
    pub fn new(tensor_weights: &SafeTensors) -> anyhow::Result<Self> {
        let mut attention_blocks = Vec::new();
        for i in 0..12 {
            attention_blocks.push(AttentionBlock::new(&tensor_weights, i)?);
        }
        Ok(Self {
            embedding_layer: EmbeddingLayer::new(&tensor_weights)?,
            attention_blocks,
        })
    }
}
