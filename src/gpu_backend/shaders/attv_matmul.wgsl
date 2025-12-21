@group(0) @binding(0) var<storage, read> attention: array<f32>;   // [n_heads, token_len, token_len]
@group(0) @binding(1) var<storage, read> v: array<f32>;           // [n_heads, token_len, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_heads, token_len, head_dim]

struct Params {
    token_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let head = global_id.z;
    let row = global_id.y;  // output token
    let col = global_id.x;  // output dimension
    
    if (head >= params.n_heads || row >= params.token_len || col >= params.head_dim) {
        return;
    }
    
    // Compute: attention[head, row, :] @ V[head, :, col]
    var sum = 0.0;
    let attn_offset = head * params.token_len * params.token_len + row * params.token_len;
    
    for (var k = 0u; k < params.token_len; k++) {
        let v_idx = head * params.token_len * params.head_dim + k * params.head_dim + col;
        sum += attention[attn_offset + k] * v[v_idx];
    }
    
    let out_idx = head * params.token_len * params.head_dim + row * params.head_dim + col;
    output[out_idx] = sum;
}