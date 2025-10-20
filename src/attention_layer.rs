use ndarray::{Array2, Axis};
pub struct AttentionLayer {
    mq: Array2<f32>,
    mk: Array2<f32>,
    mv: Array2<f32>,
    head_dim: usize,
}

impl AttentionLayer {
    pub fn new() -> Self {
        let dim = 32 as usize;
        let head_dim = 32 as usize;
        AttentionLayer {
            mq: Array2::zeros((dim, head_dim)),
            mk: Array2::zeros((dim, head_dim)),
            mv: Array2::zeros((dim, head_dim)),
            head_dim,
        }
    }

    fn softmax(input: &mut Array2<f32>) {
        for mut embedding in input.lanes_mut(Axis(0)) {
            let mut store = Vec::new();
            let mut sum = 0_f32;
            for value in &embedding {
                store.push(value.exp());
                sum += value.exp();
            }
            for i in 0..embedding.len() {
                embedding[i] = store[i] / sum;
            }
        }
    }

    pub fn run(self, input: Array2<f32>) -> Array2<f32> {
        let q = self.mq.dot(&input);
        let k = self.mk.dot(&input);
        let v = self.mv.dot(&input);
        let mut attention = q.dot(&k.t());
        Self::softmax(&mut attention);
        attention /= (self.head_dim as f32).sqrt();
        attention.dot(&v)
    }
}
