pub mod backend;
pub mod pipelines;
pub mod tensor;
mod test;

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
pub struct ComputeShape {
    m: u32,
    k: u32,
    n: u32,
}
