use serde::Deserialize;
use std::fs;
use tiktoken_rs::r50k_base;

use safetensors::tensor::SafeTensors;

use crate::{attention_block::AttentionBlock, embedding_layer::EmbeddingLayer};

mod attention_block;
mod attention_layer;
mod embedding_layer;
mod layer_norm;
mod linear_layer;
mod tools;

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

fn main() -> anyhow::Result<()> {
    let config = Config::default();
    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    let embedding_layer = EmbeddingLayer::new(&tensor_weights)?;
    let embeddings = embedding_layer.run(&tokens);

    let first_block = AttentionBlock::new(&tensor_weights, 0)?;
    let output = first_block.run(&embeddings);
    println!("{}", tokens[0]);
    println!("{}", embeddings.row(0));
    println!("{}", output.row(0));
    Ok(())
}

#[derive(Deserialize, Debug)]
struct Embeddings {
    #[serde(rename = "Token embeddings")]
    token_embeddings: Vec<f32>,
    #[serde(rename = "Position embeddings")]
    position_embeddings: Vec<f32>,
    #[serde(rename = "Combined embeddings")]
    combined_embeddings: Vec<f32>,
}

#[test]
fn test_embedding() -> anyhow::Result<()> {
    let data = fs::read_to_string("test/test_data.dump").unwrap();
    let emb: Embeddings = serde_json::from_str(&data).unwrap();

    let config = Config::default();
    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    let embedding_layer = EmbeddingLayer::new(&tensor_weights)?;
    let embeddings = embedding_layer.run(&tokens);
    assert_eq!(
        embeddings.row(0),
        ndarray::Array1::from(emb.combined_embeddings)
    );
    Ok(())
}
