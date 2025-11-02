use ndarray::Array1;
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
#[cfg(test)]
mod test;
mod tools;

fn argmax(array: &Array1<f32>) -> usize {
    let mut max_value = array[0];
    let mut max_index = 0;
    for (i, v) in array.iter().enumerate() {
        if *v > max_value {
            max_index = i;
            max_value = *v;
        }
    }
    max_index
}

fn main() -> anyhow::Result<()> {
    let bytes = fs::read("model.safetensors")?;
    let tensor_weights = SafeTensors::deserialize(&bytes)?;

    let tokenizer = r50k_base()?;
    // tokenizer.special_tokens()
    let mut tokens = tokenizer.encode_with_special_tokens("Once upon");
    let model = GPT2::new(&tensor_weights)?;
    for _ in 0..5 {
        let embeddings = model.embedding_layer.run(&tokens);
        let full_out = model.run(&embeddings)?;
        println!("{:?}", argmax(&full_out));
        tokens.push(argmax(&full_out) as u32);
    }
    println!("{:?}", tokenizer.decode(tokens));
    Ok(())
}
