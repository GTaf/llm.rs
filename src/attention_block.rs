use std::f32;

use ndarray::Array2;
use safetensors::SafeTensors;

use crate::layer_norm::LayerNorm;
use crate::{attention_layer::AttentionLayer, linear_layer::LinearLayer};

fn gelu(x: &f32) -> f32 {
    let alpha = x + 0.044715_f32 * x.powi(3);
    0.5_f32 * x * (1. + (f32::consts::FRAC_2_PI.sqrt() * alpha).tanh())
}

pub struct AttentionBlock {
    layer_norm1: LayerNorm,
    attention_layer: AttentionLayer,
    linear_proj: LinearLayer,
    layer_norm2: LayerNorm,
    linear_1: LinearLayer,
    linear_2: LinearLayer,
}

impl AttentionBlock {
    pub fn new(tensor_weights: SafeTensors, index: usize) -> anyhow::Result<Self> {
        let layer_norm_weights_1 = tensor_weights.tensor(&format!("h.{index}.ln_1.weight"))?;
        let layer_norm_bias_1 = tensor_weights.tensor(&format!("h.{index}.ln_1.bias"))?;
        let layer_norm_weights_2 = tensor_weights.tensor(&format!("h.{index}.ln_2.weight"))?;
        let layer_norm_bias_2 = tensor_weights.tensor(&format!("h.{index}.ln_2.bias"))?;

        let linproj_weights = tensor_weights.tensor(&format!("h.{index}.attn.c_proj.weight"))?;
        let linproj_bias = tensor_weights.tensor(&format!("h.{index}.attn.c_proj.bias"))?;

        let mlp_weights_1 = tensor_weights.tensor(&format!("h.{index}.mlp.c_fc.weight"))?;
        let mlp_bias_1 = tensor_weights.tensor(&format!("h.{index}.mlp.c_fc.bias"))?;

        let mlp_weights_proj = tensor_weights.tensor(&format!("h.{index}.mlp.c_proj.weight"))?;
        let mlp_bias_proj = tensor_weights.tensor(&format!("h.{index}.mlp.c_proj.bias"))?;

        Ok(Self {
            layer_norm1: LayerNorm::new(layer_norm_weights_1, layer_norm_bias_1)?,
            attention_layer: AttentionLayer::new(),
            linear_proj: LinearLayer::new(linproj_weights, linproj_bias)?,
            layer_norm2: LayerNorm::new(layer_norm_weights_2, layer_norm_bias_2)?,
            linear_1: LinearLayer::new(mlp_weights_1, mlp_bias_1)?,
            linear_2: LinearLayer::new(mlp_weights_proj, mlp_bias_proj)?,
        })
    }

    pub fn run(self, input: &Array2<f32>) -> Array2<f32> {
        let mut step = self.layer_norm1.run(input);
        step = self.attention_layer.run(step);
        step = self.linear_proj.run(step);
        step = self.layer_norm2.run(&step);
        step = self.linear_1.run(step);
        step = step.map(gelu);
        step = self.linear_2.run(step);
        step
    }
}
