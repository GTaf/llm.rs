struct Shape {
    M : u32,
    K : u32,
    N : u32,
    _pad2: u32,
}

const workgroup_len : u32 = 16;

var<workgroup> tile_input: array<f32, workgroup_len * workgroup_len>;
var<workgroup> tile_weights: array<f32, workgroup_len * workgroup_len>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> shape: Shape;

fn erf_approx(x: f32) -> f32 {
    // Abramowitz and Stegun approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    
    let sign = select(-1.0, 1.0, x >= 0.0);
    let x_abs = abs(x);
    
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x_abs * x_abs);
    
    return sign * y;
}

fn gelu_exact(x: f32) -> f32 {
    let sqrt_2 = 1.4142135623730951;
    return 0.5 * x * (1.0 + erf_approx(x / sqrt_2));
}

fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608; // sqrt(2/pi)
    let coeff = 0.044715;
    
    let x_cubed = x * x * x;
    let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    let tanh_inner = tanh(inner);
    
    return 0.5 * x * (1.0 + tanh_inner);
}

@compute
@workgroup_size(workgroup_len, workgroup_len)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let i = gid.x;
    let j = gid.y;

    if (i >= shape.M || j >= shape.K) {
        return;
    }

    let idx = i * shape.K + j;
    output[idx] = gelu_exact(input[idx]);
}