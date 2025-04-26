#include <metal_stdlib>
using namespace metal;

// Simple element-wise matrix multiply: C = A·B,
// where each thread computes one C[row,col].
kernel void matrixMultiply(
    device const float* A        [[ buffer(0) ]],
    device const float* B        [[ buffer(1) ]],
    device       float* C        [[ buffer(2) ]],
    constant     uint &N         [[ buffer(3) ]],
    uint          gid            [[ thread_position_in_grid ]]
) {
    uint row = gid / N;
    uint col = gid % N;

    float sum = 0.0f;
    // dot-product of A[row,*] and B[*,col]
    for (uint k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
