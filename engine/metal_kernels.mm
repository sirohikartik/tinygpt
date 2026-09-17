#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "metal_kernels.h"
#include <iostream>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <cstring>

namespace {

static const char* METAL_SHADERS_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// GEMV for M = 1: A (1 x K) * B (K x N) -> C (1 x N)
// Vectorized unroll over K for memory coalescing and high arithmetic intensity
kernel void kernel_gemv(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < N) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        uint k = 0;
        for (; k + 8 <= K; k += 8) {
            sum0 += A[k + 0] * B[(k + 0) * N + gid];
            sum1 += A[k + 1] * B[(k + 1) * N + gid];
            sum0 += A[k + 2] * B[(k + 2) * N + gid];
            sum1 += A[k + 3] * B[(k + 3) * N + gid];
            sum0 += A[k + 4] * B[(k + 4) * N + gid];
            sum1 += A[k + 5] * B[(k + 5) * N + gid];
            sum0 += A[k + 6] * B[(k + 6) * N + gid];
            sum1 += A[k + 7] * B[(k + 7) * N + gid];
        }
        for (; k < K; ++k) {
            sum0 += A[k] * B[k * N + gid];
        }
        C[gid] = sum0 + sum1;
    }
}

// Elementwise Vector Addition: C = A + B
kernel void kernel_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        C[gid] = A[gid] + B[gid];
    }
}

// Broadcast Bias Addition: out[row, col] = in[row, col] + bias[col]
kernel void kernel_add_bias(
    device const float* in [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < cols && gid.y < rows) {
        out[gid.y * cols + gid.x] = in[gid.y * cols + gid.x] + bias[gid.x];
    }
}

// Row-wise Numerically Stable Softmax
// Grid: (rows) threadgroups, each of size tg_size (power of 2, <= 256)
kernel void kernel_softmax(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float s_shared[256];

    // Find row maximum
    float local_max = -1e30f;
    for (uint col = tid; col < cols; col += tg_size) {
        local_max = max(local_max, in[row * cols + col]);
    }
    s_shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_shared[tid] = max(s_shared[tid], s_shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = s_shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exponentiate and sum
    float local_sum = 0.0f;
    for (uint col = tid; col < cols; col += tg_size) {
        float e = exp(in[row * cols + col] - row_max);
        out[row * cols + col] = e;
        local_sum += e;
    }
    s_shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_shared[tid] += s_shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / (s_shared[0] > 0.0f ? s_shared[0] : 1e-12f);

    // Normalize
    for (uint col = tid; col < cols; col += tg_size) {
        out[row * cols + col] *= inv_sum;
    }
}

// Row-wise Layer Normalization
// out = gamma * (x - mean) / sqrt(var + eps) + beta
kernel void kernel_layernorm(
    device const float* in [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float s_shared[256];

    // Compute mean
    float local_sum = 0.0f;
    for (uint col = tid; col < cols; col += tg_size) {
        local_sum += in[row * cols + col];
    }
    s_shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_shared[tid] += s_shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = s_shared[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute variance
    float local_var = 0.0f;
    for (uint col = tid; col < cols; col += tg_size) {
        float diff = in[row * cols + col] - mean;
        local_var += diff * diff;
    }
    s_shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_shared[tid] += s_shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = s_shared[0] / float(cols);
    float inv_std = rsqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize and scale with gamma/beta
    for (uint col = tid; col < cols; col += tg_size) {
        float norm = (in[row * cols + col] - mean) * inv_std;
        out[row * cols + col] = gamma[col] * norm + beta[col];
    }
}

// GELU Activation Function
kernel void kernel_gelu(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float x = in[gid];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[gid] = 0.5f * x * (1.0f + precise::tanh(inner));
    }
}

// Elementwise Scalar Scaling: out = in * scale
kernel void kernel_scale(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        out[gid] = in[gid] * scale;
    }
}

// Causal Masking: if col > row, out = mask_val (-1e9), else out = in
kernel void kernel_mask(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant float& mask_val [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < cols && gid.y < rows) {
        uint idx = gid.y * cols + gid.x;
        if (gid.x > gid.y) {
            out[idx] = mask_val;
        } else {
            out[idx] = in[idx];
        }
    }
}

// Matrix Transpose: out[col, row] = in[row, col]
kernel void kernel_transpose(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x < cols && gid.y < rows) {
        out[gid.x * rows + gid.y] = in[gid.y * cols + gid.x];
    }
}
)";

// Global Metal device & pipeline state singleton
struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> psoGemv = nil;
    id<MTLComputePipelineState> psoAdd = nil;
    id<MTLComputePipelineState> psoAddBias = nil;
    id<MTLComputePipelineState> psoSoftmax = nil;
    id<MTLComputePipelineState> psoLayernorm = nil;
    id<MTLComputePipelineState> psoGelu = nil;
    id<MTLComputePipelineState> psoScale = nil;
    id<MTLComputePipelineState> psoMask = nil;
    id<MTLComputePipelineState> psoTranspose = nil;

    bool initialized = false;
    std::mutex initMutex;
};

static MetalState g_metal;

// Thread-local buffer pool for zero allocation overhead per thread across OpenMP threads
struct ThreadBuffers {
    id<MTLBuffer> bufA = nil;
    id<MTLBuffer> bufB = nil;
    id<MTLBuffer> bufC = nil;
    id<MTLBuffer> bufExtra1 = nil;
    id<MTLBuffer> bufExtra2 = nil;

    id<MTLBuffer> getBuffer(id<MTLBuffer>& buf, size_t bytes) {
        if (!buf || [buf length] < bytes) {
            size_t allocSize = std::max(bytes, (size_t)65536);
            buf = [g_metal.device newBufferWithLength:allocSize options:MTLResourceStorageModeShared];
        }
        return buf;
    }

    id<MTLBuffer> getA(size_t bytes) { return getBuffer(bufA, bytes); }
    id<MTLBuffer> getB(size_t bytes) { return getBuffer(bufB, bytes); }
    id<MTLBuffer> getC(size_t bytes) { return getBuffer(bufC, bytes); }
    id<MTLBuffer> getExtra1(size_t bytes) { return getBuffer(bufExtra1, bytes); }
    id<MTLBuffer> getExtra2(size_t bytes) { return getBuffer(bufExtra2, bytes); }
};

// Global weight buffer cache to avoid copying weights between CPU and GPU on every token
struct BufferCache {
    std::unordered_map<const void*, id<MTLBuffer>> cache;
    std::shared_mutex rw_mtx;

    id<MTLBuffer> get_or_create(id<MTLDevice> dev, const void* ptr, size_t bytes) {
        {
            std::shared_lock<std::shared_mutex> rlock(rw_mtx);
            auto it = cache.find(ptr);
            if (it != cache.end()) {
                return it->second;
            }
        }
        std::unique_lock<std::shared_mutex> wlock(rw_mtx);
        auto it = cache.find(ptr);
        if (it != cache.end()) {
            return it->second;
        }
        id<MTLBuffer> buf = [dev newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
        cache[ptr] = buf;
        return buf;
    }
};

static BufferCache g_weight_cache;
static thread_local ThreadBuffers tl_buffers;

static inline uint choose_tg_size(size_t cols) {
    if (cols <= 32) return 32;
    if (cols <= 64) return 64;
    if (cols <= 128) return 128;
    return 256;
}

} // namespace

namespace metal_backend {

bool init() {
    std::lock_guard<std::mutex> lock(g_metal.initMutex);
    if (g_metal.initialized) return true;

    @autoreleasepool {
        g_metal.device = MTLCreateSystemDefaultDevice();
        if (!g_metal.device) {
            std::cerr << "[Metal Backend] Error: No Metal-capable GPU found.\n";
            return false;
        }

        g_metal.commandQueue = [g_metal.device newCommandQueue];
        if (!g_metal.commandQueue) {
            std::cerr << "[Metal Backend] Error: Failed to create MTLCommandQueue.\n";
            return false;
        }

        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:METAL_SHADERS_SOURCE];
        g_metal.library = [g_metal.device newLibraryWithSource:src options:nil error:&error];
        if (!g_metal.library) {
            std::cerr << "[Metal Backend] Shader compilation error: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        auto load_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [g_metal.library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) {
                std::cerr << "[Metal Backend] Missing function: " << name << "\n";
                return nil;
            }
            NSError* psoErr = nil;
            id<MTLComputePipelineState> pso = [g_metal.device newComputePipelineStateWithFunction:fn error:&psoErr];
            if (!pso) {
                std::cerr << "[Metal Backend] PSO creation error for " << name << ": "
                          << [[psoErr localizedDescription] UTF8String] << "\n";
            }
            return pso;
        };

        g_metal.psoGemv      = load_pso("kernel_gemv");
        g_metal.psoAdd       = load_pso("kernel_add");
        g_metal.psoAddBias   = load_pso("kernel_add_bias");
        g_metal.psoSoftmax   = load_pso("kernel_softmax");
        g_metal.psoLayernorm = load_pso("kernel_layernorm");
        g_metal.psoGelu      = load_pso("kernel_gelu");
        g_metal.psoScale     = load_pso("kernel_scale");
        g_metal.psoMask      = load_pso("kernel_mask");
        g_metal.psoTranspose = load_pso("kernel_transpose");

        if (!g_metal.psoGemv || !g_metal.psoAdd || !g_metal.psoAddBias ||
            !g_metal.psoSoftmax || !g_metal.psoLayernorm || !g_metal.psoGelu ||
            !g_metal.psoScale || !g_metal.psoMask || !g_metal.psoTranspose) {
            return false;
        }

        g_metal.initialized = true;
        std::cout << "[Metal Backend] Successfully initialized on "
                  << [[g_metal.device name] UTF8String] << " with native MPS & GPU compute kernels.\n";
        return true;
    }
}

bool is_available() {
    if (!g_metal.initialized) {
        return init();
    }
    return true;
}

void matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytesA = M * K * sizeof(float);
        size_t bytesB = K * N * sizeof(float);
        size_t bytesC = M * N * sizeof(float);

        id<MTLBuffer> bufA = tl_buffers.getA(bytesA);
        id<MTLBuffer> bufB = (bytesB >= 65536)
                                ? g_weight_cache.get_or_create(g_metal.device, B, bytesB)
                                : tl_buffers.getB(bytesB);
        id<MTLBuffer> bufC = tl_buffers.getC(bytesC);

        std::memcpy(bufA.contents, A, bytesA);
        if (bytesB < 65536) {
            std::memcpy(bufB.contents, B, bytesB);
        }

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];

        if (M == 1) {
            // Optimized GEMV path for vector-matrix multiply
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_metal.psoGemv];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            uint uK = static_cast<uint>(K);
            uint uN = static_cast<uint>(N);
            [enc setBytes:&uK length:sizeof(uint) atIndex:3];
            [enc setBytes:&uN length:sizeof(uint) atIndex:4];

            NSUInteger tg = 256;
            if (N < tg) tg = ((N + 31) / 32) * 32;
            if (tg < 32) tg = 32;
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        } else {
            // Native MPSMatrixMultiplication for M > 1
            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                              columns:K
                                                                             rowBytes:K * sizeof(float)
                                                                             dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                              columns:N
                                                                             rowBytes:N * sizeof(float)
                                                                             dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                              columns:N
                                                                             rowBytes:N * sizeof(float)
                                                                             dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

            MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc] initWithDevice:g_metal.device
                                                                              transposeLeft:NO
                                                                             transposeRight:NO
                                                                                 resultRows:M
                                                                              resultColumns:N
                                                                            interiorColumns:K
                                                                                      alpha:1.0f
                                                                                       beta:0.0f];
            [gemm encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        }

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(C, bufC.contents, bytesC);
    }
}

void add(const float* A, const float* B, float* C, size_t size) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytes = size * sizeof(float);
        id<MTLBuffer> bufA = tl_buffers.getA(bytes);
        id<MTLBuffer> bufB = tl_buffers.getB(bytes);
        id<MTLBuffer> bufC = tl_buffers.getC(bytes);

        std::memcpy(bufA.contents, A, bytes);
        std::memcpy(bufB.contents, B, bytes);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoAdd];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        uint uCount = static_cast<uint>(size);
        [enc setBytes:&uCount length:sizeof(uint) atIndex:3];

        NSUInteger tg = 256;
        [enc dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(C, bufC.contents, bytes);
    }
}

void add_bias(const float* in, const float* bias, float* out, size_t rows, size_t cols) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytesIn = rows * cols * sizeof(float);
        size_t bytesBias = cols * sizeof(float);

        id<MTLBuffer> bufIn = tl_buffers.getA(bytesIn);
        id<MTLBuffer> bufBias = (bytesBias >= 65536)
                                    ? g_weight_cache.get_or_create(g_metal.device, bias, bytesBias)
                                    : tl_buffers.getB(bytesBias);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytesIn);

        std::memcpy(bufIn.contents, in, bytesIn);
        if (bytesBias < 65536) {
            std::memcpy(bufBias.contents, bias, bytesBias);
        }

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoAddBias];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufBias offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        uint uRows = static_cast<uint>(rows);
        uint uCols = static_cast<uint>(cols);
        [enc setBytes:&uRows length:sizeof(uint) atIndex:3];
        [enc setBytes:&uCols length:sizeof(uint) atIndex:4];

        MTLSize grid = MTLSizeMake(cols, rows, 1);
        NSUInteger tgX = std::min((size_t)cols, (size_t)256);
        NSUInteger tgY = std::max((size_t)1, 256 / tgX);
        if (tgY > rows) tgY = rows;
        MTLSize tg = MTLSizeMake(tgX, tgY, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytesIn);
    }
}

void softmax(const float* in, float* out, size_t rows, size_t cols) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytes = rows * cols * sizeof(float);
        id<MTLBuffer> bufIn = tl_buffers.getA(bytes);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytes);

        std::memcpy(bufIn.contents, in, bytes);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoSoftmax];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        uint uCols = static_cast<uint>(cols);
        [enc setBytes:&uCols length:sizeof(uint) atIndex:2];

        uint tg_size = choose_tg_size(cols);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytes);
    }
}

void layernorm(const float* in, const float* gamma, const float* beta, float* out,
               size_t rows, size_t cols, float eps) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytesIn = rows * cols * sizeof(float);
        size_t bytesParam = cols * sizeof(float);

        id<MTLBuffer> bufIn = tl_buffers.getA(bytesIn);
        id<MTLBuffer> bufGamma = tl_buffers.getB(bytesParam);
        id<MTLBuffer> bufBeta = tl_buffers.getExtra1(bytesParam);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytesIn);

        std::memcpy(bufIn.contents, in, bytesIn);
        std::memcpy(bufGamma.contents, gamma, bytesParam);
        std::memcpy(bufBeta.contents, beta, bytesParam);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoLayernorm];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufGamma offset:0 atIndex:1];
        [enc setBuffer:bufBeta offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        uint uCols = static_cast<uint>(cols);
        [enc setBytes:&uCols length:sizeof(uint) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];

        uint tg_size = choose_tg_size(cols);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytesIn);
    }
}

void gelu(const float* in, float* out, size_t size) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytes = size * sizeof(float);
        id<MTLBuffer> bufIn = tl_buffers.getA(bytes);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytes);

        std::memcpy(bufIn.contents, in, bytes);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoGelu];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        uint uCount = static_cast<uint>(size);
        [enc setBytes:&uCount length:sizeof(uint) atIndex:2];

        NSUInteger tg = 256;
        [enc dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytes);
    }
}

void scale(const float* in, float* out, size_t size, float scale_val) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytes = size * sizeof(float);
        id<MTLBuffer> bufIn = tl_buffers.getA(bytes);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytes);

        std::memcpy(bufIn.contents, in, bytes);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoScale];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBytes:&scale_val length:sizeof(float) atIndex:2];
        uint uCount = static_cast<uint>(size);
        [enc setBytes:&uCount length:sizeof(uint) atIndex:3];

        NSUInteger tg = 256;
        [enc dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytes);
    }
}

void mask(const float* in, float* out, size_t rows, size_t cols, float mask_val) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytes = rows * cols * sizeof(float);
        id<MTLBuffer> bufIn = tl_buffers.getA(bytes);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytes);

        std::memcpy(bufIn.contents, in, bytes);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoMask];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        uint uRows = static_cast<uint>(rows);
        uint uCols = static_cast<uint>(cols);
        [enc setBytes:&uRows length:sizeof(uint) atIndex:2];
        [enc setBytes:&uCols length:sizeof(uint) atIndex:3];
        [enc setBytes:&mask_val length:sizeof(float) atIndex:4];

        MTLSize grid = MTLSizeMake(cols, rows, 1);
        NSUInteger tgX = std::min((size_t)cols, (size_t)256);
        NSUInteger tgY = std::max((size_t)1, 256 / tgX);
        if (tgY > rows) tgY = rows;
        MTLSize tg = MTLSizeMake(tgX, tgY, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytes);
    }
}

void transpose(const float* in, float* out, size_t rows, size_t cols) {
    if (!is_available()) return;

    @autoreleasepool {
        size_t bytes = rows * cols * sizeof(float);
        id<MTLBuffer> bufIn = tl_buffers.getA(bytes);
        id<MTLBuffer> bufOut = tl_buffers.getC(bytes);

        std::memcpy(bufIn.contents, in, bytes);

        id<MTLCommandBuffer> cb = [g_metal.commandQueue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_metal.psoTranspose];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        uint uRows = static_cast<uint>(rows);
        uint uCols = static_cast<uint>(cols);
        [enc setBytes:&uRows length:sizeof(uint) atIndex:2];
        [enc setBytes:&uCols length:sizeof(uint) atIndex:3];

        MTLSize grid = MTLSizeMake(cols, rows, 1);
        NSUInteger tgX = std::min((size_t)cols, (size_t)16);
        NSUInteger tgY = std::min((size_t)rows, (size_t)16);
        if (tgX < 1) tgX = 1;
        if (tgY < 1) tgY = 1;
        MTLSize tg = MTLSizeMake(tgX, tgY, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, bufOut.contents, bytes);
    }
}

} // namespace metal_backend
