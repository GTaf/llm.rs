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
mod test;
mod tools;

fn main() -> anyhow::Result<()> {
    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    let tokens =
        tokenizer.encode_with_special_tokens("The main character of The lord of the rings is ");
    let model = GPT2::new(&tensor_weights)?;
    let embeddings = model.embedding_layer.run(&tokens);
    let output = model.attention_blocks.get(0).unwrap().run(&embeddings)?;
    println!("{}", tokens[0]);
    println!("{}", embeddings.row(0));
    println!("{}", output.row(0));
    Ok(())
}
