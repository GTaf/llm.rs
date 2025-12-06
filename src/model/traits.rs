use async_trait::async_trait;
use ndarray::Array1;
use wgpu::{Device, Queue};

pub enum BackendType {
    Cpu,
    Gpu,
}

pub struct ModelContext {
    pub device: Device,
    pub queue: Queue,
    pub backend: BackendType,
}

#[async_trait]
pub trait LanguageModel {
    async fn run(&self, input: &[u32]) -> anyhow::Result<Array1<f32>>;
    fn encode(&self, input: String) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, input: &[u32]) -> anyhow::Result<String>;
}
