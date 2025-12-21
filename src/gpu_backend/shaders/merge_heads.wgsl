@group(0) @binding(0) var<storage, read> input: array<f32>;       // [n_heads, token_len, head_dim]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [token_len, embed_dim]

struct Params {
    token_len: u32,
    n_heads: u32,
    head_dim: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.token_len * params.n_heads * params.head_dim;
    
    if (idx >= total) {
        return;
    }
    
    // Decode output indices: idx = token * embed_dim + head * head_dim + dim
    let token = idx / (params.n_heads * params.head_dim);
    let remainder = idx % (params.n_heads * params.head_dim);
    let head = remainder / params.head_dim;
    let dim = remainder % params.head_dim;
    
    // Source index: [head, token, dim]
    let src_idx = head * params.token_len * params.head_dim + token * params.head_dim + dim;
    
    output[idx] = input[src_idx];
}