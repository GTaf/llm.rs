use rusty_gpt2::run_model;
use std::fs;

fn main() {
    let input = String::from("Once upon a time");
    let model_weights = fs::read("model.safetensors").unwrap();
    println!("{}", model_weights.len());
    let _ = run_model(input, &model_weights);
}
