@group(0) @binding(0) var<storage, read> qkv: array<f32>;        // [token_len, 3 * embed_dim]
@group(0) @binding(1) var<storage, read_write> q: array<f32>;    // [n_heads, token_len, head_dim]
@group(0) @binding(2) var<storage, read_write> k: array<f32>;    // [n_heads, token_len, head_dim]
@group(0) @binding(3) var<storage, read_write> v: array<f32>;    // [n_heads, token_len, head_dim]

struct Params {
    token_len: u32,
    n_heads: u32,
    head_dim: u32,
    embed_dim: u32,
}

@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.token_len * params.n_heads * params.head_dim;
    
    if (idx >= total) {
        return;
    }
    
    // Decode indices: idx = head * token_len * head_dim + token * head_dim + dim
    let head = idx / (params.token_len * params.head_dim);
    let remainder = idx % (params.token_len * params.head_dim);
    let token = remainder / params.head_dim;
    let dim = remainder % params.head_dim;
    
    // Source indices in QKV (before split and transpose)
    let qkv_base = token * params.embed_dim * 3u + head * params.head_dim + dim;
    
    // Copy and transpose: [token, head, dim] -> [head, token, dim]
    q[idx] = qkv[qkv_base];
    k[idx] = qkv[qkv_base + params.embed_dim];
    v[idx] = qkv[qkv_base + 2u * params.embed_dim];
}