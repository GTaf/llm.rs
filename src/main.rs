use ndarray::Array1;
use std::collections::HashMap;
use tiktoken_rs::r50k_base;

struct EmbeddingLayer {
    table: HashMap<u32, Vec<u32>>,
}

impl EmbeddingLayer {
    pub fn new() -> Self {
        EmbeddingLayer {
            table: HashMap::new(),
        }
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
            pos_encoding.push((p as f32) / (10000.0_f32).powf(2. * i as f32 / 768.));
            pos_encoding.push((p as f32) / (10000.0_f32).powf(2. * (i + 1) as f32 / 768.));
        }
        let odd_add = Array1::from_vec(pos_encoding);
        embedding.scaled_add(1., &odd_add);
    }
}

fn main() {
    let _config = Config::default();

    let tokenizer = r50k_base().unwrap();
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    println!("{}", tokens.len());
    println!("{}", tokens[0]);
}
