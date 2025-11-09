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
@workgroup_size(1)
fn main(
    // global_invocation_id specifies our position in the invocation grid
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let i = gid.x;
    let j = gid.y;

    // debug[i] = 1;


    if (i >= shape.M || j >= shape.N) {
        return;
    }

    let idx = i * shape.N + j;
    output[idx] = 0.;
    for (var k: u32 = 0; k < shape.K; k++) {
        output[idx] += input[i * shape.K + k] * weights[k * shape.N + j];
        // if(i == 1 && j == 0) {
        //     debug[2*k] = input[i * shape.M + k];
        //     debug[2*k + 1] = weights[k * shape.K + j];
        //     let offset = 2 * shape.K;
        //     debug[2*k + offset] = f32(i * shape.M + k);
        //     debug[2*k + 1 + offset] = f32(k * shape.K + j);
        // }
    }
    output[idx] += bias[j];
}
