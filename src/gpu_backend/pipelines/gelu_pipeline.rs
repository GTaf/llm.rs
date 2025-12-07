use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::{
    gpu_backend::{ComputeShape, backend::GpuBackend},
    layers::traits::Shape,
};

pub struct GeLUComputePipeline {
    pipeline: wgpu::ComputePipeline,
    backend: Arc<GpuBackend>,
}

impl GeLUComputePipeline {
    pub fn new_pipeline(backend: Arc<GpuBackend>) -> Self {
        let shader = backend
            .device
            .create_shader_module(wgpu::include_wgsl!("../gelu.wgsl"));

        let pipeline = backend
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Introduction Compute Pipeline"),
                layout: None,
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                cache: Default::default(),
            });

        Self {
            pipeline,
            backend: backend.clone(),
        }
    }

    pub async fn compute(
        &self,
        input: &wgpu::Buffer,
        shape: &Shape,
    ) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        self.compute_timestamp(input, None, shape).await
    }

    pub async fn compute_timestamp(
        &self,
        input_buffer: &wgpu::Buffer,
        timestamp: Option<&mut f64>,
        input_shape: &Shape,
    ) -> anyhow::Result<(wgpu::Buffer, Shape)> {
        let device = &self.backend.device;
        let queue = &self.backend.queue;

        let shape = ComputeShape {
            m: input_shape.rows as u32,
            k: input_shape.columns as u32,
            n: 0,
        };

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: input_buffer.size(),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let shape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shape uniform"),
            contents: bytemuck::bytes_of(&shape),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shape_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(shape.m.div_ceil(16), shape.k.div_ceil(16), 1);
        }

        queue.submit([encoder.finish()]);

        Ok((output_buffer, input_shape.clone()))
    }
}
