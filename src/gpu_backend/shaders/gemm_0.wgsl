struct Shape {
    M : u32,
    K : u32,
    N : u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
// @group(0) @binding(5) var<storage, read_write> debug: array<f32>;

@compute
@workgroup_size(16 * 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let i = gid.x;
    let j = gid.y;

    if (i >= shape.M || j >= shape.N) {
        return;
    }

    let idx = i * shape.N + j;
    output[idx] = 0.;
    for (var k: u32 = 0; k < shape.K; k++) {
        output[idx] += input[i * shape.K + k] * weights[k * shape.N + j];
    }
    output[idx] += bias[j];
}
