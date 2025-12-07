use std::f32;
use std::sync::Arc;

use flume::bounded;
use ndarray::Array2;
use safetensors::SafeTensors;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu_backend::backend::{self, GpuBackend, gpu_buffer_to_array2};
use crate::layers::gelu::Gelu;
use crate::layers::layer_norm::LayerNorm;
use crate::layers::linear_layer::GpuLinearLayer;
use crate::layers::traits::{Layer, Shape, Tensor, TensorData};
use crate::{
    layers::linear_layer::CpuLinearLayer, layers::linear_layer::LinearLayer,
    self_attention::SelfAttention,
};

pub struct AttentionBlock {
    pub layer_norm1: Box<dyn Layer>,
    pub attention_layer: SelfAttention,
    pub layer_norm2: Box<dyn Layer>,
    pub linear_1: Box<dyn Layer>,
    pub gelu: Box<dyn Layer>,
    pub linear_2: Box<dyn Layer>,
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
            linear_1: Box::new(if let Some(ref bck) = gpu_backend {
                LinearLayer::Gpu(GpuLinearLayer::new(bck.clone(), mlp_weights_1, mlp_bias_1)?)
            } else {
                LinearLayer::Cpu(CpuLinearLayer::new(mlp_weights_1, mlp_bias_1)?)
            }),
            linear_2: Box::new(if let Some(ref bck) = gpu_backend {
                LinearLayer::Gpu(GpuLinearLayer::new(
                    bck.clone(),
                    mlp_weights_proj,
                    mlp_bias_proj,
                )?)
            } else {
                LinearLayer::Cpu(CpuLinearLayer::new(mlp_weights_proj, mlp_bias_proj)?)
            }),
            gelu: Box::new(Gelu::new(gpu_backend)?),
        })
    }

    pub async fn run(
        &self,
        input: &Array2<f32>,
        backend: Option<Arc<GpuBackend>>,
    ) -> anyhow::Result<Array2<f32>> {
        let mut step = self.layer_norm1.run_cpu(input)?;
        step = self.attention_layer.run(&step).await?;
        step += input;
        let step_2 = self.layer_norm2.run_cpu(&step)?;

        let step_2: Tensor = if let Some(back) = backend.clone() {
            let shape = Shape::from(&step_2);
            let input_buffer = back.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("input_buffer"),
                contents: bytemuck::cast_slice(step_2.as_slice().unwrap()),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            });
            Tensor::new_gpu(input_buffer, shape)
        } else {
            Tensor::new_cpu(step_2)
        };

        let step_2 = self.linear_1.run(step_2).await?;
        let step_2 = self.gelu.run(step_2).await?;
        let step_2 = self.linear_2.run(step_2).await?;

        let step_2 = match backend {
            Some(backend) => {
                let shape = step_2.shape().clone();
                gpu_buffer_to_array2(backend.clone().as_ref(), step_2.data_gpu_move(), shape)
                    .await?
            }
            None => step_2.data_cpu_move(),
        };

        Ok(step_2 + step)
    }
}
