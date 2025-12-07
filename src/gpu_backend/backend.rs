use bytemuck::{Pod, Zeroable};
use flume::bounded;
use ndarray::Array2;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use web_sys::console;

use wgpu::{Buffer, Device, Queue};

use crate::{gpu_backend::ComputeShape, layers::traits::Shape};

unsafe impl Zeroable for ComputeShape {}
unsafe impl Pod for ComputeShape {}

pub struct GpuBackend {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl GpuBackend {
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();
        let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        Ok(GpuBackend { device, queue })
    }
}

pub async fn gpu_buffer_to_array2(
    backend: &GpuBackend,
    buffer: Buffer,
    shape: Shape,
) -> anyhow::Result<Array2<f32>> {
    let temp_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("temp"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = backend.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&buffer, 0, &temp_buffer, 0, buffer.size());
    backend.queue.submit([encoder.finish()]);

    let (tx, rx) = bounded(1);
    temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap()
    });

    backend.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv_async().await??;

    let range = temp_buffer.get_mapped_range(..);
    let raw_data: &[f32] = bytemuck::cast_slice(&range);
    Ok(Array2::from_shape_vec(
        (shape.rows as usize, shape.columns as usize),
        raw_data.to_vec(),
    )?)
}
