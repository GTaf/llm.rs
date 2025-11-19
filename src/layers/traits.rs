use ndarray::Array2;

pub trait Layer: Send + Sync {
    fn run(&self, input: &Array2<f32>) -> Array2<f32>;
}