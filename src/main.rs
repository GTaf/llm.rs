use ndarray::Array1;
use std::{collections::HashMap, fs};
use tiktoken_rs::r50k_base;

use safetensors::tensor::SafeTensors;

mod attention_layer;
mod linear_layer;
use crate::attention_layer::AttentionLayer;

struct EmbeddingLayer {
    table: HashMap<u32, Vec<u32>>,
    dimension: u32,
}

impl EmbeddingLayer {
    pub fn new(dimension: u32) -> Self {
        EmbeddingLayer {
            table: HashMap::new(),
            dimension,
        }
    }

    fn run(self, index: &[u32]) -> Vec<Array1<f32>> {
        let mut result = Vec::new();
        for _ in 0..index.len() {
            result.push(Array1::zeros(self.dimension as usize));
        }
        result
    }
}

struct Config {
    vocab_size: u32,
    context_length: u32,
    emb_dim: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
        }
    }
}

fn add_positionnal_embedding(config: Config, embeddings: &mut [Array1<f32>]) {
    for (p, embedding) in embeddings.iter_mut().enumerate() {
        let mut pos_encoding = Vec::new();
        for i in 0..config.emb_dim / 2 {
            pos_encoding.push(((p as f32) / (10000.0_f32).powf(2. * i as f32 / 768.)).sin());
            pos_encoding.push(((p as f32) / (10000.0_f32).powf(2. * (i + 1) as f32 / 768.)).cos());
        }
        let odd_add = Array1::from_vec(pos_encoding);
        embedding.scaled_add(1., &odd_add);
    }
}

fn load_weights() {
    let bytes = fs::read("model.safetensors").unwrap();
    let safetensor = SafeTensors::deserialize(&bytes).unwrap();
    println!("{:?}", safetensor.names());
}

fn main() {
    let config = Config::default();

    let tokenizer = r50k_base().unwrap();
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    let embedding_layer = EmbeddingLayer::new(config.emb_dim);
    let mut embeddings = embedding_layer.run(&tokens);
    add_positionnal_embedding(config, &mut embeddings);

    load_weights();

    println!("{}", tokens.len());
    println!("{}", tokens[0]);
    println!("{}", embeddings[5]);
}
