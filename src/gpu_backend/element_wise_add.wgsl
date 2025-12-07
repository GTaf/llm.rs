struct Shape {
    M : u32,
    K : u32,
    N : u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let idx = row * shape.K + col;

    if (row < shape.M && col < shape.K) {
        output[idx] = input1[idx] + input2[idx];
    }
}