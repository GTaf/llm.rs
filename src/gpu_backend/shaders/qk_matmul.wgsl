@group(0) @binding(0) var<storage, read> q: array<f32>;          // [n_heads, token_len, head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;          // [n_heads, token_len, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads, token_len, token_len]

struct Params {
    token_len: u32,
    n_heads: u32,
    head_dim: u32,
    scale: f32,  // 1.0 / sqrt(head_dim)
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let head = global_id.z;
    let row = global_id.y;  // query token
    let col = global_id.x;  // key token
    
    if (head >= params.n_heads || row >= params.token_len || col >= params.token_len) {
        return;
    }
    
    // Compute dot product: Q[head, row, :] @ K[head, col, :]^T
    var sum = 0.0;
    let q_offset = head * params.token_len * params.head_dim + row * params.head_dim;
    let k_offset = head * params.token_len * params.head_dim + col * params.head_dim;
    
    for (var d = 0u; d < params.head_dim; d++) {
        sum += q[q_offset + d] * k[k_offset + d];
    }
    
    // Scale and store
    let out_idx = head * params.token_len * params.token_len + row * params.token_len + col;
    scores[out_idx] = sum * params.scale;
}