pub enum TensorWeightsConfig {
    RawBytes(Vec<u8>),
    Path(String),
}

pub struct ModelConfig {
    pub qwen: bool,
    pub tensor_weights: TensorWeightsConfig,
    pub use_gpu: bool,
}
