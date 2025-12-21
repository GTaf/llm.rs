@group(0) @binding(0) var<storage, read_write> scores: array<f32>; // [n_heads, token_len, token_len]
@group(0) @binding(1) var<storage, read> mask: array<f32>;         // [token_len, token_len]

struct Params {
    token_len: u32,
    n_heads: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_rows = params.n_heads * params.token_len;
    
    if (idx >= total_rows) {
        return;
    }
    
    let head = idx / params.token_len;
    let row = idx % params.token_len;
    
    let base = head * params.token_len * params.token_len + row * params.token_len;
    
    // Apply mask and find max (for numerical stability)
    var max_val = -1e10;
    for (var col = 0u; col < params.token_len; col++) {
        let mask_val = mask[row * params.token_len + col];
        if (mask_val == 0.0) {
            scores[base + col] = -1e10;  // Masked positions
        }
        max_val = max(max_val, scores[base + col]);
    }
    
    // Compute exp and sum
    var sum = 0.0;
    for (var col = 0u; col < params.token_len; col++) {
        let val = exp(scores[base + col] - max_val);
        scores[base + col] = val;
        sum += val;
    }
    
    // Normalize
    for (var col = 0u; col < params.token_len; col++) {
        scores[base + col] /= sum;
    }
}