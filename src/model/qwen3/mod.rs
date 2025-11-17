use std::sync::Arc;

use async_trait::async_trait;
use ndarray::{Array1};
use safetensors::SafeTensors;
use tokenizers::tokenizer::{Tokenizer};

use crate::{
    attention_block::AttentionBlock,
    embedding_layer::EmbeddingLayer,
    gpu_backend::backend::GpuBackend,
    layer_norm::LayerNorm,
    linear_layer::{CpuLinearLayer, LinearLayer}, model::LanguageModel,
};

use half::f16;

pub struct Qwen3 {
    pub embedding_layer: EmbeddingLayer,
    pub attention_blocks: Vec<AttentionBlock>,
    pub layer_norm: LayerNorm,
    pub linear_layer: LinearLayer,
    tokenizer: Tokenizer,
}

impl Qwen3 {
    pub async fn new(tensor_weights: &SafeTensors<'_>, use_gpu: bool) -> anyhow::Result<Self> {
        let mut attention_blocks = Vec::new();
        let gpu_backend = if use_gpu {
            Some(Arc::new(GpuBackend::new().await?))
        } else {
            None
        };
        for i in 0..12 {
            attention_blocks.push(AttentionBlock::new(tensor_weights, i, gpu_backend.clone())?);
        }
        let tokenizer = Tokenizer::from_file("src/model/gpt2/tokenizer.json").unwrap();
        Ok(Self {
            embedding_layer: EmbeddingLayer::new(tensor_weights, None, "embed_tokens.weight")?,
            attention_blocks,
            layer_norm: LayerNorm::new(
                tensor_weights.tensor("ln_f.weight")?,
                tensor_weights.tensor("ln_f.bias")?,
            )?,
            linear_layer: LinearLayer::Cpu(CpuLinearLayer::new_no_bias(
                tensor_weights.tensor("wte.weight")?,
            )?),
            tokenizer
        })
    }

    pub async fn run(&self, input: &[u32]) -> anyhow::Result<Array1<f32>> {
        let embedding = self.embedding_layer.run(input);
        let mut res = embedding;
        for i in 0..12 {
            res = self.attention_blocks[i].run(&res).await?;
        }
        res = self.layer_norm.run(&res);
        let res = self
            .linear_layer
            .run(&res)
            .await?
            .row(res.shape()[0] - 1)
            .to_owned();
        Ok(res)
    }
}

#[async_trait]
impl LanguageModel for Qwen3 {
    async fn run(&self, input: &[u32]) -> anyhow::Result<Array1<f32>> {
        self.run(input).await
    }

    fn encode(&self, input: String) -> anyhow::Result<Vec<u32>> {
        // Use the input parameter instead of hardcoded string
        let tokens = Vec::from(self.tokenizer.encode(input, true).unwrap().get_ids());
        Ok(tokens)
    }
    fn decode(&self, input: &[u32]) -> anyhow::Result<String> {
        Ok(self.tokenizer.decode(input, false).unwrap())
    }
}