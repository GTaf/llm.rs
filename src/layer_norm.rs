use ndarray::{Array1, Array2};
use safetensors::tensor::TensorView;

pub struct LayerNorm {
    bias: Array1<f32>,
    weight: Array1<f32>,
}

fn weights_to_array1<'a>(tensor: &TensorView<'a>) -> Array1<f32> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats = bytemuck::cast_slice(&aligned);

    Array1::from_shape_vec((tensor.shape()[0]), floats.to_vec())
        .expect("Unable to create Array from vec")
}

impl LayerNorm {
    pub fn new(weights: TensorView, bias: TensorView) -> Self {
        Self {
            bias: weights_to_array1(&bias),
            weight: weights_to_array1(&weights),
        }
    }

    pub fn run(self, input: Array2<f32>) -> Array2<f32> {
        todo!();
    }
}
