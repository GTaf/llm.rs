use bytemuck::{Pod, Zeroable};
use flume::bounded;
#[cfg(test)]
use pollster::FutureExt;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use web_sys::console;

use ndarray::{Array1, Array2};
use wgpu::{
    Buffer, Device, Queue,
    util::{BufferInitDescriptor, DeviceExt},
};

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct Shape {
    m: u32,
    k: u32,
    n: u32,
}

unsafe impl Zeroable for Shape {}
unsafe impl Pod for Shape {}

pub struct GpuBackend {
    pub device: Arc<Device>,
    queue: Arc<Queue>,
}

pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    weights: Buffer,
    bias: Buffer,
    backend: Arc<GpuBackend>,
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
            .request_device(&&wgpu::DeviceDescriptor {
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

impl ComputePipeline {
    pub fn new_pipeline(
        backend: Arc<GpuBackend>,
        weights: Array2<f32>,
        bias: Array1<f32>,
    ) -> ComputePipeline {
        let shader = backend
            .device
            .create_shader_module(wgpu::include_wgsl!("gemm.wgsl"));

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

        let weights = backend.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(weights.as_slice().unwrap()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let bias = backend.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("bias"),
            contents: bytemuck::cast_slice(bias.as_slice().unwrap()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        ComputePipeline {
            pipeline,
            weights,
            bias,
            backend: backend.clone(),
        }
    }

    pub async fn compute(&self, input: Array2<f32>) -> anyhow::Result<Array2<f32>> {
        self.compute_timestamp(input, None).await
    }

    pub async fn compute_timestamp(
        &self,
        input: Array2<f32>,
        timestamp: Option<&mut f64>,
    ) -> anyhow::Result<Array2<f32>> {
        let device = &self.backend.device;
        let queue = &self.backend.queue;

        let shape = Shape {
            m: input.shape()[0] as u32,
            k: input.shape()[1] as u32,
            n: self.bias.size() as u32 / 4,
        };

        let input_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("input_buffer"),
            contents: bytemuck::cast_slice(input.as_slice().unwrap()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (shape.m * shape.n * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // let debug_buffer_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("debug"),
        //     size: 128,
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        //     mapped_at_creation: false,
        // });
        // let debug_buffer_cpu = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("debug"),
        //     size: 128,
        //     usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });
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
                    resource: self.weights.as_entire_binding(),
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
                // wgpu::BindGroupEntry {
                //     binding: 5,
                //     resource: debug_buffer_gpu.as_entire_binding(),
                // },
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

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &temp_buffer, 0, output_buffer.size());
        // encoder.copy_buffer_to_buffer(&debug_buffer_gpu, 0, &debug_buffer_cpu, 0, 128);

        queue.submit([encoder.finish()]);

        // The mapping process is async, so we'll need to create a channel to get
        // the success flag for our mapping
        let (tx, rx) = bounded(1);
        // let (tx1, rx1) = bounded(1);

        // We send the success or failure of our mapping via a callback
        temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });

        device.poll(wgpu::PollType::wait_indefinitely())?; // We check if the mapping was successful here
        rx.recv_async().await??;

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

        // debug_buffer_cpu.map_async(wgpu::MapMode::Read, .., move |result| {
        //     tx1.send(result).unwrap()
        // });
        // device.poll(wgpu::PollType::wait_indefinitely())?;
        // rx1.recv_async().await??;

        // We then get the bytes that were stored in the buffer
        let output_data = temp_buffer.get_mapped_range(..);
        // let debug_data = debug_buffer_cpu.get_mapped_range(..);

        let raw_data: &[f32] = bytemuck::cast_slice(&output_data);
        // let debug_data: &[f32] = bytemuck::cast_slice(&debug_data);

        // We need to unmap the buffer to be able to use it again
        // temp_buffer.unmap();
        let output =
            Array2::from_shape_vec((shape.m as usize, shape.n as usize), raw_data.to_vec())
                .unwrap();
        Ok(output)
    }
}

#[test]
fn test_gpu_backend_2x2() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec((2, 2), vec![1_f32, 2_f32, 3_f32, 4_f32])?;
    let bias = Array1::from_shape_vec(2, vec![1_f32, 2_f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec((2, 2), vec![1_f32, 2_f32, 3_f32, 4_f32])?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec((2, 2), vec![8_f32, 12_f32, 16_f32, 24_f32])?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_3x3_1() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 1_f32, 2_f32,
        ],
    )?;
    let bias = Array1::from_shape_vec(3, vec![0_f32, 0_f32, 0f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32,
        ],
    )?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec(
            (3, 3),
            vec![
                4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32
            ]
        )?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_3x3_2() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 2_f32, 1_f32,
        ],
    )?;
    let bias = Array1::from_shape_vec(3, vec![0_f32, 0_f32, 0f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 0_f32, 0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0_f32, 1_f32,
        ],
    )?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec(
            (3, 3),
            vec![
                1_f32, 2_f32, 3_f32, 2_f32, 3_f32, 1_f32, 3_f32, 2_f32, 1_f32,
            ]
        )?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_3x3_3() -> anyhow::Result<()> {
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (3, 3),
        vec![
            1_f32, 0_f32, 0_f32, 0_f32, 1_f32, 0_f32, 0_f32, 0_f32, 1_f32,
        ],
    )?;
    let bias = Array1::from_shape_vec(3, vec![0_f32, 0_f32, 0f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights, bias);
    let input = Array2::from_shape_vec(
        (3, 3),
        vec![
            4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32,
        ],
    )?;
    let output = compute_pipeline.compute(input).block_on()?;
    assert_eq!(
        output,
        Array2::from_shape_vec(
            (3, 3),
            vec![
                4_f32, 3_f32, 5_f32, 2_f32, 3_f32, 1_f32, 4_f32, 3_f32, 5_f32
            ]
        )?
    );

    Ok(())
}

#[test]
fn test_gpu_backend_4x4_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (4, 4),
        (0..16).map(|_| rng.gen_range(0., 4.)).collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(4, vec![0_f32, 0_f32, 0f32, 0_f32])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (4, 4),
        (0..16).map(|_| rng.gen_range(0., 4.)).collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_32x32_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let matrix_size = 32;
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (matrix_size, matrix_size),
        (0..matrix_size * matrix_size)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(matrix_size, vec![0_f32; matrix_size])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (matrix_size, matrix_size),
        (0..matrix_size * matrix_size)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_3_2_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (3, 2, 1);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_32_10_11_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (32, 10, 11);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_32_32_32_1() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (32, 32, 32);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_16_16_16() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (16, 16, 16);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}

#[test]
fn test_gpu_backend_33() -> anyhow::Result<()> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let (m, k, n) = (33, 33, 33);
    let backend = Arc::new(GpuBackend::new().block_on()?);
    let weights = Array2::from_shape_vec(
        (k, n),
        (0..k * n)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;

    let bias = Array1::from_shape_vec(n, vec![0_f32; n])?;
    let compute_pipeline = ComputePipeline::new_pipeline(backend.clone(), weights.clone(), bias);
    let input = Array2::from_shape_vec(
        (m, k),
        (0..m * k)
            .map(|_| rng.gen_range(0., 4.))
            .collect::<Vec<f32>>(),
    )?;
    let output = compute_pipeline.compute(input.clone()).block_on()?;
    assert_eq!(output, input.dot(&weights));

    Ok(())
}
