use std::sync::Arc;

use ndarray::{Array1, Array2};
use wgpu::{
    Buffer,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    gpu_backend::{ComputeShape, backend::GpuBackend, tensor::Tensor},
    layers::traits::Shape,
};

pub struct LinearComputePipeline {
    pipeline: wgpu::ComputePipeline,
    weights: Tensor,
    bias: Buffer,
    backend: Arc<GpuBackend>,
}

impl LinearComputePipeline {
    pub fn new_pipeline(
        backend: Arc<GpuBackend>,
        weights: Array2<f32>,
        bias: Array1<f32>,
    ) -> LinearComputePipeline {
        let shader = backend
            .device
            .create_shader_module(wgpu::include_wgsl!("../gemm.wgsl"));

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

        let wshape = Shape::from(&weights);
        let weights = backend.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(weights.as_slice().unwrap()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });
        let weights = Tensor::new_gpu(weights, wshape);

        let bias = backend.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("bias"),
            contents: bytemuck::cast_slice(bias.as_slice().unwrap()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        LinearComputePipeline {
            pipeline,
            weights,
            bias,
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
            n: self.weights.shape().columns as u32,
        };

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (shape.m * shape.n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("query resolve buffer"),
            size: size_of::<u64>() as u64 * 2,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });
        let dest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("query dest buffer"),
            size: size_of::<u64>() as u64 * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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
                    resource: self.weights.data_gpu().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shape_buffer.as_entire_binding(),
                },
            ],
        });
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp query set"),
            count: 2,
            ty: wgpu::QueryType::Timestamp,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.write_timestamp(&query_set, 0);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(shape.m.div_ceil(16), shape.n.div_ceil(16), 1);
        }
        encoder.write_timestamp(&query_set, 1);
        encoder.resolve_query_set(&query_set, 0..2, &resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&resolve_buffer, 0, &dest_buffer, 0, resolve_buffer.size());

        queue.submit([encoder.finish()]);

        dest_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let timestamps: Vec<u64> = {
            let timestamp_view = dest_buffer
                .slice(..(size_of::<u64>() as wgpu::BufferAddress * 2))
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };
        if let Some(ts) = timestamp {
            *ts = (timestamps[1] - timestamps[0]) as f64 * queue.get_timestamp_period() as f64;
        }

        let output_shape = Shape {
            columns: shape.n as usize,
            rows: shape.m as usize,
        };
        Ok((output_buffer, output_shape))
    }
}
