use std::f32;
use std::sync::Arc;

use ndarray::Array2;
use safetensors::SafeTensors;

use crate::gpu_backend::backend::GpuBackend;
use crate::layers::layer_norm::LayerNorm;
use crate::linear_layer::GpuLinearLayer;
use crate::{
    linear_layer::CpuLinearLayer, linear_layer::LinearLayer, self_attention::SelfAttention,
};
use crate::layers::traits::Layer;

pub fn gelu(x: &f32) -> f32 {
    // 0.5 * x * (1.0 + ((2.0 / f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    // x * normal_cdf(x)
    0.5 * x * (1.0 + libm::erff(x / 2.0_f32.sqrt()))
}

pub struct AttentionBlock {
    pub layer_norm1: Box<dyn Layer>,
    pub attention_layer: SelfAttention,
    pub layer_norm2: Box<dyn Layer>,
    pub linear_1: LinearLayer,
    pub linear_2: LinearLayer,
}

pub enum NormType {
    RMSNorm(String),
    LayerNorm((String, String)),
}

pub struct AttentionConfig {
    norm_type: NormType,
}

impl AttentionBlock {
    pub fn new(
        tensor_weights: &SafeTensors,
        index: usize,
        gpu_backend: Option<Arc<GpuBackend>>,
    ) -> anyhow::Result<Self> {
        let layer_norm_weights_1 = tensor_weights.tensor(&format!("h.{index}.ln_1.weight"))?;
        let layer_norm_bias_1 = tensor_weights.tensor(&format!("h.{index}.ln_1.bias"))?;
        let layer_norm_weights_2 = tensor_weights.tensor(&format!("h.{index}.ln_2.weight"))?;
        let layer_norm_bias_2 = tensor_weights.tensor(&format!("h.{index}.ln_2.bias"))?;

        let attn_weights = tensor_weights.tensor(&format!("h.{index}.attn.c_attn.weight"))?;
        let attn_bias = tensor_weights.tensor(&format!("h.{index}.attn.c_attn.bias"))?;
        let linproj_weights = tensor_weights.tensor(&format!("h.{index}.attn.c_proj.weight"))?;
        let linproj_bias = tensor_weights.tensor(&format!("h.{index}.attn.c_proj.bias"))?;

        let mlp_weights_1 = tensor_weights.tensor(&format!("h.{index}.mlp.c_fc.weight"))?;
        let mlp_bias_1 = tensor_weights.tensor(&format!("h.{index}.mlp.c_fc.bias"))?;

        let mlp_weights_proj = tensor_weights.tensor(&format!("h.{index}.mlp.c_proj.weight"))?;
        let mlp_bias_proj = tensor_weights.tensor(&format!("h.{index}.mlp.c_proj.bias"))?;

        let causal_weights = tensor_weights.tensor(&format!("h.{index}.attn.bias"))?;

        Ok(Self {
            layer_norm1: Box::new(LayerNorm::new(layer_norm_weights_1, layer_norm_bias_1)?),
            attention_layer: SelfAttention::new(
                linproj_weights,
                linproj_bias,
                attn_weights,
                attn_bias,
                causal_weights,
            )?,
            layer_norm2: Box::new(LayerNorm::new(layer_norm_weights_2, layer_norm_bias_2)?),
            linear_1: if let Some(ref bck) = gpu_backend {
                LinearLayer::Gpu(GpuLinearLayer::new(bck.clone(), mlp_weights_1, mlp_bias_1)?)
            } else {
                LinearLayer::Cpu(CpuLinearLayer::new(mlp_weights_1, mlp_bias_1)?)
            },
            linear_2: if let Some(ref bck) = gpu_backend {
                LinearLayer::Gpu(GpuLinearLayer::new(
                    bck.clone(),
                    mlp_weights_proj,
                    mlp_bias_proj,
                )?)
            } else {
                LinearLayer::Cpu(CpuLinearLayer::new(mlp_weights_proj, mlp_bias_proj)?)
            },
        })
    }

    pub async fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        let mut step = self.layer_norm1.run(input);
        step = self.attention_layer.run(&step).await?;
        step += input;
        let mut step_2 = self.layer_norm2.run(&step);
        step_2 = self.linear_1.run(&step_2).await?;
        step_2 = step_2.map(gelu);
        step_2 = self.linear_2.run(&step_2).await?;
        Ok(step_2 + step)
    }
}
