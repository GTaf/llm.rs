struct Shape {
    M : u32,
    K : u32,
    N : u32,
    _pad2: u32,
}

const workgroup_len : u32 = 16;

var<workgroup> tile_input: array<f32, workgroup_len * workgroup_len>;
var<workgroup> tile_weights: array<f32, workgroup_len * workgroup_len>;

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;
// @group(0) @binding(5) var<storage, read_write> debug: array<f32>;

@compute
@workgroup_size(workgroup_len * workgroup_len)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let step_nb = (shape.K + workgroup_len - 1) / workgroup_len;
    
    let local_i = local_invocation_id.x / workgroup_len;  // row index in workgroup_len x workgroup_len tile
    let local_j = local_invocation_id.x % workgroup_len;  // col index in workgroup_len x workgroup_len tile

    let i = workgroup_id.x * workgroup_len + local_i;
    let j = workgroup_id.y * workgroup_len + local_j;
    

    let idx = i * shape.N + j;
    let is_valid = (i < shape.M && j < shape.N);

    output[idx] = 0.;

    for(var tile_id: u32 = 0; tile_id < step_nb; tile_id++) {
        let input_i = workgroup_id.x * workgroup_len + local_i;
        let input_j = tile_id * workgroup_len + local_j;
        
        let weights_i = tile_id * workgroup_len + local_i;
        let weights_j = workgroup_id.y * workgroup_len + local_j;
        

        if (input_j < shape.K) {
            tile_input[local_invocation_id.x] = input[input_i * shape.K + input_j];
        };
        if (weights_i < shape.K) {
            tile_weights[local_invocation_id.x] = weights[weights_i * shape.N + weights_j];
        };
        workgroupBarrier();

        // Compute the valid k range for this tile
        let k_start = tile_id * workgroup_len;
        let k_end = min(k_start + workgroup_len, shape.K);
        let k_limit = k_end - k_start;
        for (var k: u32 = 0; is_valid && k < k_limit; k++) {
            output[idx] += tile_input[local_i * workgroup_len + k] * 
            tile_weights[k * workgroup_len + local_j];
        }
        
        workgroupBarrier();
    }
    output[idx] += bias[j];
}
