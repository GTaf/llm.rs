use serde::Deserialize;
use std::fs;
use tiktoken_rs::r50k_base;

use safetensors::tensor::SafeTensors;

use crate::gpt2::GPT2;

mod attention_block;
mod embedding_layer;
mod gpt2;
mod layer_norm;
mod linear_layer;
mod self_attention;
mod tools;

fn main() -> anyhow::Result<()> {
    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");
    let model = GPT2::new(&tensor_weights)?;
    let embeddings = model.embedding_layer.run(&tokens);
    let output = model.attention_blocks.get(0).unwrap().run(&embeddings);
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
    #[serde(rename = "First layer norm")]
    first_layer_norm: Vec<f32>,
}

#[test]
fn test_embedding() -> anyhow::Result<()> {
    let data = fs::read_to_string("test/test_data.dump").unwrap();
    let emb: Embeddings = serde_json::from_str(&data).unwrap();

    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    let model = GPT2::new(&tensor_weights)?;
    let embeddings = model.embedding_layer.run(&tokens);
    assert_eq!(
        embeddings.row(0),
        ndarray::Array1::from(emb.combined_embeddings)
    );
    Ok(())
}

#[test]
fn test_layer_norm() -> anyhow::Result<()> {
    let data = fs::read_to_string("test/test_data.dump").unwrap();
    let emb: Embeddings = serde_json::from_str(&data).unwrap();

    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");

    let model = GPT2::new(&tensor_weights)?;
    let embeddings = model.embedding_layer.run(&tokens);
    let output = model
        .attention_blocks
        .get(0)
        .unwrap()
        .layer_norm1
        .run(&embeddings);
    let tested_row = output.row(0);
    for i in 0..tested_row.len() {
        let rust_val = tested_row[i];
        let py_val = emb.first_layer_norm[i];
        let rel_error = ((rust_val - py_val) / py_val.abs().max(1e-8)).abs();

        // This is necessary because of f32 imprecision doing the calculus of mean or var
        assert!(
            rel_error < 1e-4, // Tolérance de 0.01%
            "Mismatch at [{}]: {} vs {} (rel error: {:.2e})",
            i,
            rust_val,
            py_val,
            rel_error
        );
    }
    Ok(())
}
