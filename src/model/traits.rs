use ndarray::Array2;

trait LanguageModel {
    fn run(&self, input: &Array2<f32>) -> anyhow::Result<Array2<f32>>;
}