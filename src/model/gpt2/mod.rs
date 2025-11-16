use std::sync::Arc;

use async_trait::async_trait;
use ndarray::{Array1};
use safetensors::SafeTensors;

use crate::{
    attention_block::AttentionBlock,
    embedding_layer::EmbeddingLayer,
    gpu_backend::backend::GpuBackend,
    layer_norm::LayerNorm,
    linear_layer::{CpuLinearLayer, LinearLayer}, model::LanguageModel,
};

pub struct GPT2 {
    pub embedding_layer: EmbeddingLayer,
    pub attention_blocks: Vec<AttentionBlock>,
    pub layer_norm: LayerNorm,
    pub linear_layer: LinearLayer,
}

impl GPT2 {
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
        Ok(Self {
            embedding_layer: EmbeddingLayer::new(tensor_weights)?,
            attention_blocks,
            layer_norm: LayerNorm::new(
                tensor_weights.tensor("ln_f.weight")?,
                tensor_weights.tensor("ln_f.bias")?,
            )?,
            linear_layer: LinearLayer::Cpu(CpuLinearLayer::new_no_bias(
                tensor_weights.tensor("wte.weight")?,
            )?),
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
impl LanguageModel for GPT2 {
    async fn run(&self, input: &[u32]) -> anyhow::Result<Array1<f32>> {
        self.run(input).await
    }
}