use llmrs::run_model;
use pollster::FutureExt;
use std::fs;

fn main() {
    let input = String::from("Once upon a time,");
    let model_weights = fs::read("model.safetensors").unwrap();
    let _ = run_model(input, &model_weights).block_on();
}
