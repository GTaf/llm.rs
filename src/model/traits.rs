use async_trait::async_trait;
use ndarray::Array1;

#[async_trait]
pub trait LanguageModel {
    async fn run(&self, input: &[u32]) -> anyhow::Result<Array1<f32>>;
    fn encode(&self, input: String) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, input: &[u32]) -> anyhow::Result<String>;
}
