use llmrs::run_model;
use pollster::FutureExt;
use std::fs;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Use gpu or not
    #[arg(short, long, default_value_t = false)]
    use_gpu: bool,
}

fn main() {
    let args = Args::parse();

    let input = String::from("Once upon a time,");
    let model_weights = fs::read("model.safetensors").unwrap();
    let _ = run_model(input, &model_weights, args.use_gpu).block_on();
}
