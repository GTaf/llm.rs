use std::fs;

use safetensors::SafeTensors;

use crate::model::{
    LanguageModel,
    config::{ModelConfig, TensorWeightsConfig},
    gpt2::GPT2,
    qwen3::Qwen3,
};

pub struct ModelBuilder {
    config: ModelConfig,
}

impl ModelBuilder {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }

    pub async fn build(self) -> Box<dyn LanguageModel> {
        let model_weights = match self.config.tensor_weights {
            TensorWeightsConfig::RawBytes(raw) => raw,
            TensorWeightsConfig::Path(path) => fs::read(path).unwrap(),
        };

        let tensor_weights = SafeTensors::deserialize(&model_weights).unwrap();
        match &self.config.qwen {
            false => Box::new(
                GPT2::new(&tensor_weights, self.config.use_gpu)
                    .await
                    .unwrap(),
            ),
            true => Box::new(
                Qwen3::new(&tensor_weights, self.config.use_gpu)
                    .await
                    .unwrap(),
            ),
        }
    }
}
