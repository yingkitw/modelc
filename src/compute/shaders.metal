#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel
// Each thread computes one element of the result matrix C = A * B
kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;

    if (i >= m || j >= n) return;

    float sum = 0.0f;
    for (uint p = 0; p < k; p++) {
        sum += A[i * k + p] * B[p * n + j];
    }
    C[i * n + j] = sum;
}

// Optimized matrix multiplication with shared memory
kernel void matmul_kernel_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint TILE_SIZE = 16;

    uint row = gid.x;
    uint col = gid.y;

    float sum = 0.0f;

    uint tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < tiles; t++) {
        // Load tiles into shared memory
        uint a_col = t * TILE_SIZE + tid.y;
        uint b_row = t * TILE_SIZE + tid.x;

        if (row < m && a_col < k) {
            shared_A[tid.x * TILE_SIZE + tid.y] = A[row * k + a_col];
        } else {
            shared_A[tid.x * TILE_SIZE + tid.y] = 0.0f;
        }

        if (b_row < k && col < n) {
            shared_B[tid.x * TILE_SIZE + tid.y] = B[b_row * n + col];
        } else {
            shared_B[tid.x * TILE_SIZE + tid.y] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product using shared memory
        for (uint p = 0; p < TILE_SIZE; p++) {
            sum += shared_A[tid.x * TILE_SIZE + p] * shared_B[p * TILE_SIZE + tid.y];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// Softmax kernel along specified axis
kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant uint& outer [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint o = gid.x;
    uint i = gid.y;

    if (o >= outer || i >= inner) return;

    uint base = o * dim * inner + i;

    // Find max
    float max_val = input[base];
    for (uint d = 1; d < dim; d++) {
        float val = input[base + d * inner];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint d = 0; d < dim; d++) {
        float val = exp(input[base + d * inner] - max_val);
        output[base + d * inner] = val;
        sum += val;
    }

    // Normalize
    for (uint d = 0; d < dim; d++) {
        output[base + d * inner] /= sum;
    }
}

// Layer normalization kernel
kernel void layer_norm_kernel(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& last_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& n_vectors [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;

    if (i >= n_vectors || j >= last_dim) return;

    uint base = i * last_dim;

    // Compute mean (first pass)
    float mean = 0.0f;
    for (uint d = 0; d < last_dim; d++) {
        mean += input[base + d];
    }
    mean /= last_dim;

    // Compute variance and normalize (second pass)
    float var = 0.0f;
    for (uint d = 0; d < last_dim; d++) {
        float diff = input[base + d] - mean;
        var += diff * diff;
    }
    var /= last_dim;

    float inv_std = 1.0f / sqrt(var + eps);
    output[base + j] = (input[base + j] - mean) * inv_std * weight[j] + bias[j];
}

// ReLU activation kernel
kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = fmax(0.0f, input[gid]);
}

// Element-wise addition kernel
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = a[gid] + b[gid];
}

// Scalar multiplication kernel
kernel void mul_scalar_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    output[gid] = input[gid] * scalar;
}