use ndarray::Array1;
#[cfg(test)]
use pollster::FutureExt;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::fs;
use tokenizers::tokenizer::Tokenizer;

use crate::{layers::Layer, model::gpt2::GPT2};

#[derive(Deserialize, Debug)]
struct Embeddings {#[serde(rename = "Raw tokens")]
    raw_tokens: Vec<u32>,
    #[serde(rename = "Token embeddings")]
    _token_embeddings: Vec<f32>,
    #[serde(rename = "Position embeddings")]
    _position_embeddings: Vec<f32>,
    #[serde(rename = "Combined embeddings")]
    combined_embeddings: Vec<f32>,
    #[serde(rename = "First layer norm")]
    first_layer_norm: Vec<f32>,
    #[serde(rename = "First layer attention-exp")]
    first_layer_attention_exp: Vec<f32>,
    #[serde(rename = "First layer attention")]
    first_layer_attention: Vec<f32>,
    #[serde(rename = "First layer norm 2")]
    first_layer_norm2: Vec<f32>,
    #[serde(rename = "First layer full")]
    first_layer_full: Vec<f32>,
    #[serde(rename = "First layer mlp 1")]
    first_layer_mlp1: Vec<f32>,

    #[serde(rename = "First layer gelu")]
    first_layer_gelu: Vec<f32>,
    #[serde(rename = "First layer mlp 2")]
    first_layer_mlp2: Vec<f32>,
    #[serde(rename = "First layer full manual")]
    first_layer_full_manual: Vec<f32>,
}

#[cfg(test)]
fn test_setup() -> anyhow::Result<(GPT2, Vec<u32>, Embeddings)> {
    let input = "The main character of The lord of the rings is ";
    let data = fs::read_to_string("test/test_data.dump").unwrap();
    let emb: Embeddings = serde_json::from_str(&data).unwrap();

    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = Tokenizer::from_file("src/model/gpt2/tokenizer.json").unwrap();
    let tokens = Vec::from(tokenizer.encode(input, true).unwrap().get_ids());

    let model = GPT2::new(&tensor_weights, true).block_on()?;
    Ok((model, tokens, emb))
}

#[cfg(test)]
fn test_proximity_threshold<'a, T, U>(a: U, b: T, threshold: f32, full_debug: bool) -> bool
where
    T: IntoIterator<Item = &'a f32>,
    U: IntoIterator<Item = &'a f32>,
{
    let mut result = true;
    for (k, (x, y)) in a.into_iter().zip(b.into_iter()).enumerate() {
        if (x - y).abs() > threshold {
            result = false;
            if full_debug {
                println!("{} : {} {}", k, x, y);
            }
        }
    }
    result
}

#[test]
fn test_token() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let first_result: Vec<u32> = emb.raw_tokens.iter().map(|u| *u as u32).collect();

    assert_eq!(
        tokens,
        first_result
    );
    Ok(())
}

#[test]
fn test_embedding() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;

    let embeddings = model.embedding_layer.run(&tokens);
    assert_eq!(
        embeddings.row(0),
        ndarray::Array1::from(emb.combined_embeddings)
    );
    Ok(())
}

#[test]
fn test_layer_norm() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let output = model
        .attention_blocks
        .get(0)
        .unwrap()
        .layer_norm1
        .run_cpu(&embeddings)?;
    let tested_row = output.row(0);
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_norm,
        1e-4,
        false
    ));

    Ok(())
}

#[test]
fn test_layer_attention_linearity() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let output = attention_block
        .attention_layer
        .linear_expand
        .run(&output)
        .block_on()?;
    let tested_row = output.row(0);
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_attention_exp,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_attention() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let output = attention_block.attention_layer.run(&output).block_on()?;
    let tested_row = output.row(0);
    println!(
        "{:?}\nrust out {:?}\n python out {:?}",
        tested_row.shape(),
        tested_row[0],
        emb.first_layer_attention[0]
    );
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_attention,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_norm2() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let output = attention_block.attention_layer.run(&output).block_on()?;
    let output = attention_block.layer_norm2.run_cpu(&(output + embeddings))?;
    let tested_row = output.row(0);
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_norm2,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_mlp_lin1() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let output = attention_block.attention_layer.run(&output).block_on()?;
    let output = attention_block.layer_norm2.run_cpu(&(output + embeddings))?;
    let mlp_output = attention_block.linear_1.run(&output).block_on()?;
    let tested_row = mlp_output.row(0);
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_mlp1,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_mlp_gelu() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let output = attention_block.attention_layer.run(&output).block_on()?;
    let output = attention_block.layer_norm2.run_cpu(&(output + embeddings))?;
    let mlp_output = attention_block.linear_1.run(&output).block_on()?;
    let mlp_output = attention_block.gelu.run_cpu(&mlp_output)?;
    let tested_row = mlp_output.row(0);

    println!(
        "{:?}\n{:?}\n{:?}",
        tested_row.shape(),
        tested_row[0],
        emb.first_layer_gelu[0]
    );
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_gelu,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_mlp_mlp2() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let output = attention_block.attention_layer.run(&output).block_on()?;
    let output = attention_block.layer_norm2.run_cpu(&(output + embeddings))?;
    let mlp_output = attention_block.linear_1.run(&output).block_on()?;
    let mlp_output = attention_block.gelu.run_cpu(&mlp_output)?;
    let mlp_output = attention_block.linear_2.run(&mlp_output).block_on()?;
    let tested_row = mlp_output.row(0);

    println!(
        "{:?}\n{:?}\n{:?}",
        tested_row.shape(),
        tested_row[0],
        emb.first_layer_mlp2[0]
    );
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_mlp2,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_mlp_full_manual() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.layer_norm1.run_cpu(&embeddings)?;
    let attention_output = attention_block.attention_layer.run(&output).block_on()?;
    let intermediate = attention_output.clone() + &embeddings;
    let norm2_output = attention_block.layer_norm2.run_cpu(&intermediate)?;
    let mlp_output = attention_block.linear_1.run(&norm2_output).block_on()?;
    let mlp_output = attention_block.gelu.run_cpu(&mlp_output)?;
    let mlp_output = attention_block.linear_2.run(&mlp_output).block_on()?;
    println!(
        "emb_rust : {:?}\n emb python : {:?}\n\nattn_out rust : {:?}\n attn out python : {:?}\n\nskip conn output rust : {:?}\n skip con python : {:?}\n\nmlp out rust : {:?}\nmlp_output_python : {:?}",
        embeddings.row(0)[0],
        &emb.combined_embeddings[0],
        attention_output.row(0)[0],
        emb.first_layer_attention[0],
        intermediate.row(0)[0],
        (Array1::from_vec(emb.first_layer_attention.clone())
            + Array1::from_vec(emb.combined_embeddings.clone()))[0],
        mlp_output.row(0)[0],
        emb.first_layer_mlp2[0]
    );
    let mlp_output = mlp_output + intermediate;
    let tested_row = mlp_output.row(0);

    println!(
        "{:?}\n{:?}\n{:?}",
        tested_row.shape(),
        tested_row[0],
        emb.first_layer_full_manual[0]
    );
    assert!(!test_proximity_threshold(
        tested_row,
        &emb.first_layer_full_manual,
        1e-4,
        false
    ));
    Ok(())
}

#[test]
fn test_layer_full() -> anyhow::Result<()> {
    let (model, tokens, emb) = test_setup()?;
    let embeddings = model.embedding_layer.run(&tokens);
    let attention_block = model.attention_blocks.get(0).unwrap();
    let output = attention_block.run(&embeddings).block_on()?;
    let tested_row = output.row(0);
    println!(
        "Rust {:?}\nPython : {:?}",
        tested_row[0], emb.first_layer_full[0]
    );
    assert!(test_proximity_threshold(
        tested_row,
        &emb.first_layer_full,
        1e-1,
        false
    ));
    Ok(())
}
