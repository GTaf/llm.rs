use ndarray::Array2;
use safetensors::tensor::TensorView;

pub struct EmbeddingLayer {
    dimension: usize,
    wpe_f32: Array2<f32>,
    wte_f32: Array2<f32>,
}

fn weights_to_array<'a>(tensor: &TensorView<'a>) -> Array2<f32> {
    let bytes = tensor.data();
    let mut aligned = Vec::<u8>::with_capacity(bytes.len());
    aligned.extend_from_slice(bytes);
    let floats = bytemuck::cast_slice(&aligned);

    Array2::from_shape_vec((tensor.shape()[0], tensor.shape()[1]), floats.to_vec())
        .expect("Unable to create Array from vec")
}

impl EmbeddingLayer {
    pub fn new(dimension: usize, wte: TensorView, wpe: TensorView) -> Self {
        let wpe_f32 = weights_to_array(&wpe);
        let wte_f32 = weights_to_array(&wte);

        EmbeddingLayer {
            dimension,
            wte_f32,
            wpe_f32,
        }
    }

    pub fn run(self, tokens: &[u32]) -> Array2<f32> {
        let mut result = Array2::zeros((tokens.len(), self.dimension));
        for p in 0..tokens.len() {
            result.row_mut(p).scaled_add(1., &self.wpe_f32.row(p));
            result
                .row_mut(p)
                .scaled_add(1., &self.wte_f32.row(tokens[p] as usize));
        }
        result
    }
}
