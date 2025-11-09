use ndarray::Array1;
use safetensors::tensor::SafeTensors;
use tiktoken_rs::r50k_base;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

use crate::gpt2::GPT2;
mod attention_block;
mod embedding_layer;
mod gpt2;
mod gpu_backend;
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

pub fn run_model(input: String, model_bytes: &[u8]) -> anyhow::Result<String> {
    #[cfg(target_arch = "wasm32")]
    console::log_1(&format!("Model bytes length: {}", model_bytes.len()).into());

    let tensor_weights = SafeTensors::deserialize(model_bytes)?;

    #[cfg(target_arch = "wasm32")]
    console::log_1(&format!("Input text: {}", input).into());

    let tokenizer = r50k_base()?;

    // Use the input parameter instead of hardcoded string
    let mut tokens = tokenizer.encode_with_special_tokens(&input);
    let model = GPT2::new(&tensor_weights)?;

    for _ in 0..3 {
        let embeddings = model.embedding_layer.run(&tokens);
        let full_out = model.run(&embeddings)?;

        #[cfg(target_arch = "wasm32")]
        console::log_1(&format!("Next token: {:?}", argmax(&full_out)).into());

        tokens.push(argmax(&full_out) as u32);
    }

    let decoded = tokenizer.decode(tokens.clone())?;

    #[cfg(target_arch = "wasm32")]
    console::log_1(&format!("Decoded output: {}", decoded).into());

    #[cfg(not(target_arch = "wasm32"))]
    println!("{}", decoded);

    Ok(decoded)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen()]
pub fn run_web(input: String, model_bytes: &[u8]) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    console::log_1(&"Hello from Rust!".into());
    console::log_1(
        &format!(
            "Input: '{}', model weights size: {} bytes",
            &input,
            model_bytes.len()
        )
        .into(),
    );

    match run_model(input, model_bytes) {
        Ok(output) => {
            console::log_1(&format!("Success! Output: {}", output).into());
            Ok(output)
        }
        Err(e) => {
            let error_msg = format!("Error: {:?}", e);
            console::log_1(&error_msg.clone().into());
            Err(JsValue::from_str(&error_msg))
        }
    }
}
