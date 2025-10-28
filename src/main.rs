use ndarray::{Array1, Array2};
use std::fs;
use tiktoken_rs::r50k_base;

use safetensors::tensor::{SafeTensors, TensorView};

use crate::{attention_block::AttentionBlock, embedding_layer::EmbeddingLayer};

mod attention_block;
mod attention_layer;
mod embedding_layer;
mod layer_norm;
mod linear_layer;

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

    let embedding_layer = EmbeddingLayer::new(
        config.emb_dim,
        tensor_weights.tensor("wte.weight")?,
        tensor_weights.tensor("wpe.weight")?,
    );
    let embeddings = embedding_layer.run(&tokens);

    let first_block = AttentionBlock::new(tensor_weights, 0)?;
    // let output = first_block.run(&embeddings);
    println!("{}", tokens[0]);
    println!("{}", embeddings.row(0));
    Ok(())
}

#[test]
fn test_emebdding() {}
