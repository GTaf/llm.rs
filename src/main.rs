use ndarray::{Array1, Array2};
use std::fs;
use tiktoken_rs::r50k_base;

use safetensors::tensor::{SafeTensors, TensorView};

mod attention_layer;
mod linear_layer;

struct EmbeddingLayer {
    dimension: usize,
    wpe_f32: Array2<f32>,
    wte_f32: Array2<f32>,
}

fn weights_to_array<'a>(tensor: &TensorView<'a>) -> Array2<f32> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats = bytemuck::cast_slice(&aligned);

    Array2::from_shape_vec((tensor.shape()[0], tensor.shape()[1]), floats.to_vec())
        .expect("Unable to create Array from vec")
}

impl EmbeddingLayer {
    pub fn new(dimension: usize, wte: TensorView, wpe: TensorView) -> Self {
        println!("Shape positionnal {:?} {:?}", wpe.shape(), wpe.dtype());
        println!("Emb positionnal {:?}", wte.shape());

        // let wpe_f32: &[f32] = bytemuck::cast_slice(wpe.data());
        let wpe_f32 = weights_to_array(&wpe);

        let wte_f32 = weights_to_array(&wte);

        EmbeddingLayer {
            dimension,
            wte_f32,
            wpe_f32,
        }
    }

    fn run(self, tokens: &[u32]) -> Vec<Array1<f32>> {
        let mut result = Vec::new();
        for p in 0..tokens.len() {
            result.push(Array1::<f32>::zeros(self.dimension));
            result[p].scaled_add(1., &self.wpe_f32.row(p));
            result[p].scaled_add(1., &self.wte_f32.row(tokens[p] as usize));
        }
        result
    }
}

struct Config {
    _vocab_size: u32,
    _context_length: u32,
    emb_dim: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            _vocab_size: 50257,
            _context_length: 1024,
            emb_dim: 768,
        }
    }
}

fn load_weights(bytes: &[u8]) -> SafeTensors<'_> {
    let safetensor = SafeTensors::deserialize(bytes).unwrap();
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
        tensor_weights.tensor("wte.weight").unwrap(),
        tensor_weights.tensor("wpe.weight").unwrap(),
    );
    let embeddings = embedding_layer.run(&tokens);

    println!("{}", tokens.len());
    println!("{}", tokens[0]);
    println!("{}", embeddings[0]);
}
