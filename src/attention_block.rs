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
    attention_layer: AttentionLayer,
    layer_norm1: LayerNorm,
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
        Ok(Self {
            attention_layer: AttentionLayer::new(),
            layer_norm1: LayerNorm::new(layer_norm_weights_1, layer_norm_bias_1),
            layer_norm2: LayerNorm::new(layer_norm_weights_2, layer_norm_bias_2),
            linear_1: LinearLayer::new(768, 3072),
            linear_2: LinearLayer::new(3072, 768),
        })
    }

    fn run(self, input: Array2<f32>) {
        let mut step = self.layer_norm1.run();
        step = self.attention_layer.run(input);
        step = self.layer_norm2.run(input);
        step = self.linear_1.run(step);
        step = step.map(|x| gelu(x));
        step = self.linear_2.run(step);
    }
}
