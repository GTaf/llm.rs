use ndarray::Array1;
use rand::{Rng, thread_rng};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

use crate::model::{
    builder::ModelBuilder,
    config::{ModelConfig, TensorWeightsConfig},
};
mod attention_block;
mod embedding_layer;
pub mod gpu_backend;
mod layers;
mod model;
mod self_attention;
#[cfg(test)]
mod test;
mod tools;

// Fully Ai generated code
fn choose_token(tokens: &Array1<f32>, temperature: f32, top_k: usize, top_p: f32) -> usize {
    // Apply temperature and create indexed values
    let mut indexed_values: Vec<(usize, f32)> = tokens
        .indexed_iter()
        .map(|(i, &val)| (i, val / temperature))
        .collect();

    // Sort by logits in descending order
    indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Apply top-k filtering
    if top_k > 0 && top_k < indexed_values.len() {
        indexed_values.truncate(top_k);
    }

    // Convert logits to probabilities using softmax
    let max_logit = indexed_values[0].1;
    let exp_values: Vec<f32> = indexed_values
        .iter()
        .map(|(_, logit)| (logit - max_logit).exp())
        .collect();
    let sum_exp: f32 = exp_values.iter().sum();
    let probabilities: Vec<f32> = exp_values.iter().map(|v| v / sum_exp).collect();

    // Apply top-p (nucleus) filtering
    let mut cumulative_prob = 0.0;
    let mut nucleus_size = 0;
    for &prob in &probabilities {
        cumulative_prob += prob;
        nucleus_size += 1;
        if cumulative_prob >= top_p {
            break;
        }
    }

    // Truncate to nucleus and renormalize
    let nucleus_probs = &probabilities[..nucleus_size];
    let nucleus_sum: f32 = nucleus_probs.iter().sum();
    let normalized_probs: Vec<f32> = nucleus_probs.iter().map(|p| p / nucleus_sum).collect();

    // Sample from the distribution
    let mut rng = thread_rng();
    let random_value: f32 = rng.r#gen();

    let mut cumulative = 0.0;
    for (i, &prob) in normalized_probs.iter().enumerate() {
        cumulative += prob;
        if random_value <= cumulative {
            return indexed_values[i].0;
        }
    }

    // Fallback: return the last token in the nucleus
    indexed_values[nucleus_size - 1].0
}

pub async fn run_model(input: String, model_bytes: &[u8], use_gpu: bool) -> anyhow::Result<String> {
    #[cfg(target_arch = "wasm32")]
    console::log_1(&format!("Model bytes length: {}", model_bytes.len()).into());

    let model_config = ModelConfig {
        qwen: false,
        tensor_weights: TensorWeightsConfig::RawBytes(model_bytes.to_vec()),
        use_gpu,
    };
    let builder = ModelBuilder::from_config(model_config);
    let model = builder.build().await;
    #[cfg(target_arch = "wasm32")]
    console::log_1(&format!("Input text: {}", input).into());

    let mut tokens = model.encode(input)?;

    for _ in 0..10 {
        let full_out = model.run(&tokens).await?;

        tokens.push(choose_token(&full_out, 0.6, 20, 0.95) as u32);
    }
    let decoded = model.decode(&tokens).unwrap();

    #[cfg(target_arch = "wasm32")]
    console::log_1(&format!("Decoded output: {}", decoded).into());

    #[cfg(not(target_arch = "wasm32"))]
    println!("{}", decoded);

    Ok(decoded)
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::js_sys;
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen()]
pub async fn run_web(input: String, model_bytes: js_sys::Uint8Array) -> js_sys::Promise {
    let model_bytes = model_bytes.to_vec();
    use wasm_bindgen_futures::future_to_promise;
    future_to_promise(async move {
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

        match run_model(input, &model_bytes, true).await {
            Ok(output) => {
                console::log_1(&format!("Success! Output: {}", output).into());
                Ok(JsValue::from_str(&output))
            }
            Err(e) => {
                let error_msg = format!("Error: {:?}", e);
                console::log_1(&error_msg.clone().into());
                Err(JsValue::from_str(&error_msg))
            }
        }
    })
}
