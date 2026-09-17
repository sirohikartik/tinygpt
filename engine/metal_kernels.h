#pragma once

#include <vector>
#include <cstddef>

namespace metal_backend {

// Initialize Metal device, command queue, and precompile compute pipelines.
// Safe to call multiple times (idempotent).
bool init();

// Check if Metal / MPS is available and initialized.
bool is_available();

// Matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N.
// Uses Apple MPSMatrixMultiplication for M > 1 and optimized Metal GEMV for M == 1.
void matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N);

// Elementwise addition: C = A + B (size elements)
void add(const float* A, const float* B, float* C, size_t size);

// Broadcast bias addition: out[i * cols + j] = in[i * cols + j] + bias[j]
void add_bias(const float* in, const float* bias, float* out, size_t rows, size_t cols);

// Row-wise Softmax: each row is normalized with numerically stable softmax
void softmax(const float* in, float* out, size_t rows, size_t cols);

// Row-wise Layer Normalization: out = gamma * (x - mean) / sqrt(var + eps) + beta
void layernorm(const float* in, const float* gamma, const float* beta, float* out,
               size_t rows, size_t cols, float eps = 1e-5f);

// GELU activation function: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu(const float* in, float* out, size_t size);

// Elementwise scalar scale: out[i] = in[i] * scale
void scale(const float* in, float* out, size_t size, float scale);

// Causal Mask: upper triangular elements (col > row) set to mask_val (-1e9)
void mask(const float* in, float* out, size_t rows, size_t cols, float mask_val = -1e9f);

// Matrix Transpose: out[j * rows + i] = in[i * cols + j]
void transpose(const float* in, float* out, size_t rows, size_t cols);

} // namespace metal_backend
