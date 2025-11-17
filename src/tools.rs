use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

use half::f16;

pub fn weights_to_array<'a>(tensor: &TensorView<'a>) -> anyhow::Result<Array2<f32>> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);

    let floats = match tensor.dtype() {
        safetensors::Dtype::F16 => {
            // Load as f16 → convert elementwise → f32
            let f16_slice: &[f16] = bytemuck::cast_slice(&aligned);
            let converted: Vec<f32> = f16_slice.iter().map(|x| x.to_f32()).collect(); converted
        },
        safetensors::Dtype::F32 => {
            let f32_slice: &[f32] = bytemuck::cast_slice(&aligned); f32_slice.to_vec()},
        _ => todo!(),
    };

    Ok(Array2::from_shape_vec(
        (tensor.shape()[0], tensor.shape()[1]),
        floats,
    )?)
}

pub fn weights_to_array_causal<'a>(tensor: &TensorView<'a>) -> anyhow::Result<Array2<f32>> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats: &[f32] = bytemuck::cast_slice(&aligned);

    Ok(Array2::from_shape_vec(
        (tensor.shape()[2], tensor.shape()[3]),
        floats.to_vec(),
    )?)
}

pub fn weights_to_array1<'a>(tensor: &TensorView<'a>) -> anyhow::Result<Array1<f32>> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats = bytemuck::cast_slice(&aligned);

    Ok(Array1::from_shape_vec(tensor.shape()[0], floats.to_vec())?)
}
