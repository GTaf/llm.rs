use ndarray::Array1;
use std::{collections::HashMap, fs, mem};
use tiktoken_rs::r50k_base;

use safetensors::tensor::{SafeTensors, TensorView};

mod attention_layer;
mod linear_layer;

struct EmbeddingLayer<'a> {
    table: HashMap<u32, Vec<u32>>,
    dimension: usize,
    wpe: TensorView<'a>,
    wte: TensorView<'a>,
    wpe_f32: Vec<f32>,
    wte_f32: Vec<f32>,
}

impl<'a> EmbeddingLayer<'a> {
    pub fn new(dimension: usize, wte: TensorView<'a>, wpe: TensorView<'a>) -> Self {
        println!("Shape positionnal {:?} {:?}", wpe.shape(), wpe.dtype());
        println!("Emb positionnal {:?}", wte.shape());

        // let wpe_f32: &[f32] = bytemuck::cast_slice(wpe.data());
        let wpe_f32: Vec<f32> = wpe
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        println!("WPE sizes : {} {:?}", wpe_f32.len(), wpe.shape());

        let wte_f32: Vec<f32> = wte
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        println!("WTE sizes : {} {:?}", wte_f32.len(), wte.shape());

        EmbeddingLayer {
            table: HashMap::new(),
            dimension,
            wte,
            wpe,
            wte_f32,
            wpe_f32,
        }
    }

    fn run(self, tokens: &[u32]) -> Vec<Array1<f32>> {
        let mut result = Vec::new();
        for p in 0..tokens.len() {
            result.push(Array1::zeros(self.dimension as usize));
            for i in 0..self.dimension {
                result[p][i] += self.wpe_f32[p * self.dimension + i];
                // result[p][i] += self.wte_f32[tokens[p] as usize];
            }
        }
        result
    }
}

struct Config {
    vocab_size: u32,
    context_length: u32,
    emb_dim: usize,
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

fn load_weights(bytes: &[u8]) -> SafeTensors {
    let safetensor = SafeTensors::deserialize(&bytes).unwrap();
    println!(
        "Shape {:?}",
        safetensor.tensor("wpe.weight").unwrap().shape()
    );
    safetensor
}

fn main() {
    let config = Config::default();
    let bytes = fs::read("model.safetensors").unwrap();
    let tensor_weights = load_weights(&bytes);

    let tokenizer = r50k_base().unwrap();
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    let embedding_layer = EmbeddingLayer::new(
        config.emb_dim,
        tensor_weights.tensor("wpe.weight").unwrap(),
        tensor_weights.tensor("wte.weight").unwrap(),
    );
    let mut embeddings = embedding_layer.run(&tokens);
    add_positionnal_embedding(config, &mut embeddings);

    println!("{}", tokens.len());
    println!("{}", tokens[0]);
    println!("{}", embeddings[0]);
}
