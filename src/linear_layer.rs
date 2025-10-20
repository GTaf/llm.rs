use ndarray::Array2;
pub struct LinearLayer {
    weight: Array2<f32>,
    bias: Array2<f32>,
    output_dim: usize,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weight: Array2::zeros((input_dim, output_dim)),
            bias: Array2::zeros((input_dim, output_dim)),
            output_dim,
        }
    }

    pub fn run(self, input: Array2<f32>) -> Array2<f32> {
        input.dot(&self.weight) + self.bias
    }
}
