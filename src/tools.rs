use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

pub fn weights_to_array<'a>(tensor: &TensorView<'a>) -> anyhow::Result<Array2<f32>> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats: &[f32] = bytemuck::cast_slice(&aligned);

    Ok(Array2::from_shape_vec(
        (tensor.shape()[0], tensor.shape()[1]),
        floats.to_vec(),
    )?)
}

pub fn weights_to_array1<'a>(tensor: &TensorView<'a>) -> Array1<f32> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats = bytemuck::cast_slice(&aligned);

    Array1::from_shape_vec(tensor.shape()[0], floats.to_vec())
        .expect("Unable to create Array from vec")
}
