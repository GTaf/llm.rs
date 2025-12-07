struct Shape {
    M : u32,  // number of sequences/rows to normalize
    K : u32,  // hidden dimension (size of each row)
    N : u32,  // unused for layernorm
    _pad2: u32,
}

const workgroup_len : u32 = 256;
const epsilon: f32 = 1e-5;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> shape: Shape;
@group(0) @binding(3) var<storage, read> weights: array<f32>;  // gamma
@group(0) @binding(4) var<storage, read> bias: array<f32>;     // beta

@compute
@workgroup_size(workgroup_len)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = workgroup_id.x * workgroup_len + local_id.x;
    
    // Early exit if out of bounds
    if (row >= shape.M) {
        return;
    }
    
    let row_len = shape.K;
    
    // Step 1: Compute sum for mean (each thread handles one or more elements)
    var mean: f32 = 0.0;
    var variance: f32 = 0.0;
    
    // Stride through the row, each thread accumulates multiple elements
    for (var i: u32 = 0; i < row_len; i += 1) {
        let val = input[row * row_len + i];
        mean += val;
    }
    mean /= f32(row_len);

    for (var i: u32 = 0; i < row_len; i += 1) {
        let val = input[row * row_len + i];
        variance += (val-mean)*(val-mean);
    }
    variance /= f32(row_len);
    
    let inv_std = 1.0 / sqrt(variance + epsilon);
    
    // Step 4: Normalize and apply affine transformation
    for (var i: u32 = 0; i < row_len; i += 1) {
        output[row * row_len + i] = (input[row * row_len + i] - mean) * inv_std * weights[i] + bias[i];
    }
}