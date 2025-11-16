use async_trait::async_trait;
use ndarray::{Array1};

#[async_trait]
pub trait LanguageModel {
    async fn run(&self, input: &[u32]) -> anyhow::Result<Array1<f32>>;
}