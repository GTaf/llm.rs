use ndarray::{Array2, Array3, Axis, s};
use safetensors::tensor::TensorView;

use crate::{
    layers::linear_layer::CpuLinearLayer, layers::linear_layer::LinearLayer, tools::weights_to_array_causal,
};
pub struct SelfAttention {
    pub linear_expand: LinearLayer,
    pub linear_project: LinearLayer,
    pub causal_mask: Array2<f32>,
    head_number: usize,
}

fn apply_causal_mask(attn_scores: &Array2<f32>, mask: &Array2<f32>) -> Array2<f32> {
    let mut result = attn_scores.clone();

    for ((i, j), &is_allowed) in mask.indexed_iter() {
        if is_allowed == 0_f32 {
            result[[i, j]] = f32::NEG_INFINITY; // ou -1e10
        }
    }
    result
}

impl SelfAttention {
    pub fn new(
        linproj_weights: TensorView,
        linproj_bias: TensorView,
        attn_weights: TensorView,
        attn_bias: TensorView,
        causal: TensorView,
    ) -> anyhow::Result<Self> {
        let head_number = 12_usize;
        Ok(SelfAttention {
            linear_expand: LinearLayer::Cpu(CpuLinearLayer::new(attn_weights, attn_bias)?),
            linear_project: LinearLayer::Cpu(CpuLinearLayer::new(linproj_weights, linproj_bias)?),
            causal_mask: weights_to_array_causal(&causal)?,
            head_number,
        })
    }

    fn softmax(input: &mut Array2<f32>) {
        for mut embedding in input.lanes_mut(Axis(1)) {
            let mut store = Vec::new();
            let mut sum = 0_f32;
            for value in &embedding {
                store.push(value.exp());
                sum += value.exp();
            }
            for i in 0..embedding.len() {
                embedding[i] = store[i] / sum;
            }
        }
    }

    pub async fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        let embeddings_dim = input.shape()[1];
        let token_len = input.shape()[0];
        let head_dim = embeddings_dim / self.head_number;
        let expanded = self.linear_expand.run(input).await?;
        let q = expanded.slice(s![.., 0..embeddings_dim]);
        let k = expanded.slice(s![.., embeddings_dim..2 * embeddings_dim]);
        let v = expanded.slice(s![.., 2 * embeddings_dim..]);
        let mut k_split = k.to_shape([token_len, self.head_number, head_dim])?;
        let mut v_split = v.to_shape([token_len, self.head_number, head_dim])?;
        let mut q_split = q.to_shape([token_len, self.head_number, head_dim])?;
        let mask_split = self.causal_mask.slice(s![0..token_len, 0..token_len]);
        q_split.swap_axes(1, 0);
        k_split.swap_axes(1, 0);
        v_split.swap_axes(1, 0);
        let mut attention = Array3::zeros((self.head_number, token_len, head_dim));
        for head in 0..self.head_number {
            let q_head = q_split.slice(s![head, .., ..]); // [token_len, head_dim]
            let k_head = k_split.slice(s![head, .., ..]); // [token_len, head_dim]
            let v_head = v_split.slice(s![head, .., ..]); // [token_len, head_dim]
            let mut attention_head = q_head.dot(&k_head.t());
            attention_head /= (head_dim as f32).sqrt();
            attention_head = apply_causal_mask(&attention_head, &mask_split.to_owned());
            Self::softmax(&mut attention_head);
            attention_head = attention_head.dot(&v_head);
            attention
                .slice_mut(s![head, .., ..])
                .assign(&attention_head);
        }

        attention.swap_axes(1, 0);
        let attention = attention.to_shape((token_len, embeddings_dim))?.to_owned();

        self.linear_project.run(&attention).await
    }
}
