use pollster::FutureExt;
use rusty_gpt2::run_model;
use std::fs;

fn main() {
    let input = String::from("Once upon a time,");
    let model_weights = fs::read("model.safetensors").unwrap();
    let _ = run_model(input, &model_weights).block_on();
}
