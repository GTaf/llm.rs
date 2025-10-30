use ndarray::{Array2, Axis, s};
use safetensors::tensor::TensorView;

use crate::linear_layer::LinearLayer;
pub struct SelfAttention {
    linear_expand: LinearLayer,
    linear_project: LinearLayer,
    head_dim: usize,
}

impl SelfAttention {
    pub fn new(
        linproj_weights: TensorView,
        linproj_bias: TensorView,
        attn_weights: TensorView,
        attn_bias: TensorView,
    ) -> anyhow::Result<Self> {
        let head_dim = 768_usize;
        Ok(SelfAttention {
            linear_expand: LinearLayer::new(attn_weights, attn_bias)?,
            linear_project: LinearLayer::new(linproj_weights, linproj_bias)?,
            head_dim,
        })
    }

    fn softmax(input: &mut Array2<f32>) {
        for mut embedding in input.lanes_mut(Axis(0)) {
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

    pub fn run(&self, input: &Array2<f32>) -> Array2<f32> {
        let embeddings_dim = input.shape()[1];
        let expanded = self.linear_expand.run(input);
        println!("{:?}", expanded.shape());
        let q = expanded.slice(s![.., 0..embeddings_dim]);
        let k = expanded.slice(s![.., embeddings_dim..2 * embeddings_dim]);
        let v = expanded.slice(s![.., 2 * embeddings_dim..]);
        println!(
            "q : {:?}    k : {:?} v : {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
        let mut attention = q.dot(&k.t());
        println!("{:?}", attention.shape());
        Self::softmax(&mut attention);
        attention /= (self.head_dim as f32).sqrt();
        println!("{:?} {:?}", attention.shape(), v.shape());
        attention = attention.dot(&v);
        println!("{:?}", attention.shape());
        self.linear_project.run(&attention)
    }
}
