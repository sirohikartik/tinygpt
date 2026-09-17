# tinygpt

Try - https://tinygpt.onrender.com

A custom high-performance C++ inference engine for transformer-based language models. This engine is designed to run a 30 million parameter model trained on the TinyStories dataset.

Cool version of this Readme -> https://sirohikartik.github.io/tinygpt/docs

## Quick Start (macOS)

This project is specifically developed for macOS Apple Silicon. To build and run:

```bash
# Step 0: Install OpenMP dependency
brew install libomp

# Step 1: Build the project
# For CPU (ARM NEON + Apple Accelerate BLAS):
make cpu

# Or for Apple Metal GPU (Apple MPS + Native MSL Compute Shaders):
make mps

# Step 2: Run inference
./a.out

# Step 3: Run CPU vs MPS comparative benchmark
./benchmark_compare.sh
```

## Platform Compatibility

This project is optimized exclusively for **macOS (Apple Silicon)**, supporting dual acceleration backends:
1. **CPU Backend**: High-performance vectorized compute utilizing **ARM NEON** intrinsics and **Apple Accelerate BLAS**.
2. **Metal GPU Backend**: Native Apple **Metal Shading Language (MSL)** compute kernels and **Metal Performance Shaders (MPS)** via Objective-C++.

---

## Apple Metal / MPS GPU Acceleration

The GPU backend introduces end-to-end hardware acceleration on Apple Silicon GPUs without relying on external frameworks like PyTorch or ONNX Runtime.

### 1. Metal Architecture & Kernel Design
Implemented in [`engine/metal_kernels.h`](engine/metal_kernels.h) and [`engine/metal_kernels.mm`](engine/metal_kernels.mm):
- **Custom Metal Shading Language (MSL) Kernels**:
  - `kernel_gemv`: Highly optimized vector-matrix multiplication for autoregressive single-token decoding ($M=1$), processing 4 elements per thread with loop unrolling.
  - `kernel_add` & `kernel_add_bias`: Vectorized 4-wide float element-wise addition and broadcasted bias summation.
  - `kernel_softmax`: Numerically stable row-wise softmax computing max reduction, exponential sum, and normalization across execution threads.
  - `kernel_layernorm`: Mean and variance reduction across hidden dimensions with affine scaling and bias.
  - `kernel_gelu`: Fast approximate GELU activation ($0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$) operating on 4-wide vectors.
  - `kernel_scale`, `kernel_mask`, `kernel_transpose`: In-place scaling, causal triangular attention masking, and 2D matrix transposition.
- **Apple Metal Performance Shaders (MPS)**:
  - Invokes `MPSMatrixMultiplication` for prompt evaluation ($M > 1$), utilizing hardware matrix units on the Apple M-series GPU.

### 2. High-Performance Driver & Memory Optimizations
- **Zero-Copy Unified Memory**: All allocations use `MTLResourceStorageModeShared`, enabling host CPU and Apple GPU cores to read/write identical physical DRAM without memory bus copies.
- **Thread-Safe Weight Buffer Cache**: Weights loaded from `.npy` files are cached in an `MTLBuffer` registry using a Reader-Writer lock (`std::shared_mutex`), preventing OpenMP thread contention during parallel attention head projections.
- **Thread-Local Scratch Buffers**: Each CPU thread maintains dedicated reusable `MTLBuffer` allocations to eliminate per-operation driver allocation stalls.
- **Unretained Command Buffers**: Dispatches use `commandBufferWithUnretainedReferences` to eliminate driver retain/release bookkeeping overhead.

### 3. Build Flags & Automatic Backend Switching
The engine cleanly switches between CPU and GPU backends using preprocessor macros:
- Compile with `make mps` (passes `-DUSE_MPS=1 -framework Metal -framework MetalPerformanceShaders -framework Foundation`).
- In [`engine/tensor.hpp`](engine/tensor.hpp) and [`engine/runner.hpp`](engine/runner.hpp), tensor operations and attention dispatch automatically to Metal/MPS implementations when `USE_MPS` is defined, and fall back to ARM NEON + BLAS otherwise.

---

## Performance Comparison

All benchmarks were measured on Apple Silicon using 5 independent runs:

| Metric | PyTorch (MPS) | Apple Metal GPU (Custom) | Custom CPU Engine (BLAS) |
| :--- | :--- | :--- | :--- |
| **Time to First Token (TTFT)** | ~910 ms | 102.7 ms | **8.3 ms** |
| **Decode Step Time** | 87.0 ms | 41.6 ms | **2.4 ms** |
| **Decode Throughput** | 11.5 tok/s | 24.0 tok/s | **422+ tok/s** |

> **Latency & Driver Overhead Analysis**:
> For small models (d_model=256, 8 heads, head_dim=32), single-token autoregressive decoding involves small tensor operations that complete in $<0.05\ \mu\text{s}$ on CPU ARM NEON registers directly inside the L1 cache. On GPU, each kernel dispatch incurs ~15–30 $\mu\text{s}$ of Metal driver scheduling, command buffer commit, and interrupt signaling overhead. As a result, CPU vectorization is significantly faster for small single-token decoding, while the Metal GPU backend excels at batched GEMMs and larger prompt prefill.

The engine achieves industry-leading CPU inference throughput on Apple Silicon through low-level hardware optimizations:

### 1. Fused In-Register NEON Attention (Decode $M=1$)
During autoregressive generation ($M=1$), standard attention pipelines incur heavy overhead from allocating intermediate matrices for $Q K^T$, transposing $K$, scaling, causal masking, softmax, and multiplying by $V$.
- **Zero Allocations & Transpose Elimination**: The fused NEON kernel directly streams the cached keys and values from contiguous memory.
- **Hardware Vectorization**: Uses ARM NEON intrinsics (`vld1q_f32`, `vfmaq_f32`, `vaddvq_f32`) to compute dot products 4 floats at a time in hardware registers.
- **In-Register Softmax & Value Accumulation**: Computes running maximum reduction, numerically stable exponentiation, and projects directly into the head output vector in a single pass on CPU cache lines.

### 2. Batched AMX / BLAS QKV Projections
- Instead of executing 24 individual small GEMM operations per layer (8 heads $\times$ 3 projections of size $1 \times 32$), weights for $Q, K, V$ are concatenated horizontally into unified $256 \times 256$ projection matrices.
- The projection evaluates in 3 unified BLAS GEMM calls, fully saturating the Apple Matrix Coprocessor (AMX) units with a **3.45× GEMM speedup**.

### 3. In-Place $O(1)$ KV Cache Growth
- Previous implementations called `concat_vertical`, which allocated and re-copied the entire history buffer on every token step ($O(T^2)$ memory copying).
- Replaced with in-place buffer growth (`append_slice` and `append_tensor`), eliminating all intermediate vector copies and heap reallocations.

### 4. Zero-Allocation Token Generation Loop
- Replaced per-token dynamic allocation of the 50,257-element vocabulary probability distribution vector with a reusable preallocated buffer hoisted outside the decode loop.

### 5. SIMD Kernels (NEON/AVX)
Utilizes hardware-level vectorization to process multiple data points in a single instruction.
- **Implemented in**: `operator+`, `operator*`, `LayerNorm`, `operator/`, `softmax`, `add_bias`.

### 6. Tiled Matrix Multiplication
Implements a cache-efficient 32x32 tiling strategy to minimize cache misses and maximize memory bandwidth utilization.
- **Implemented in**: `operator*`.

### 7. OpenMP Parallelism
Distributes independent compute-heavy loops across multiple CPU cores, using adaptive thresholds to avoid threading overhead on small tensors.
- **Implemented in**: `operator*` (matmul), `operator+` (addition), `softmax`, `add_bias`, `gelu`, and `multiheadattention`.

## Features

| Feature | Description |
|---------|-------------|
| Multi-Head Attention | Full implementation with causal masking |
| Layer Normalization | Pre-normalization architecture support |
| BPE Tokenizer | GPT-2 compatible byte-level BPE tokenization |
| Positional Encodings | Sinusoidal positional embeddings |
| Temperature Sampling | Configurable sampling with temperature scaling |

## Project Structure

```
Inference/
├── engine/
│   ├── tensor.hpp          # Tensor class with matmul, softmax, LayerNorm
│   ├── tokenizer.hpp       # GPT-2 BPE tokenizer implementation
│   ├── runner.hpp          # Model, Transformer, Runner classes
│   ├── parser.hpp          # Weight loading from .npy files
│   ├── npy.hpp             # NumPy file parser (external library)
│   └── json.hpp            # JSON parser (external library)
├── weights/
│   ├── embeddings.weight.npy
│   ├── out.weight.npy
│   ├── out.bias.npy
│   └── transforms.*.npy    # Transformer block weights
├── gpt2_vocab.json         # Tokenizer vocabulary
├── merges.txt              # BPE merge rules
├── tokenizer.py            # Python script to export tokenizer
├── export_tokenizer.py     # Alternative tokenizer export script
├── convert.py              # Model conversion utilities
├── run.cpp                 # Main entry point
├── Makefile                # Build configuration
└── README.md               # This file
```

## Model Architecture

The engine supports a decoder-only transformer with the following configuration:

| Parameter | Value |
|-----------|-------|
| Parameters | ~30M |
| d_model | 256 |
| Number of Heads | 8 |
| Number of Blocks | 6 |
| d_k (per head) | 32 |
| Vocabulary Size | 50257 |
| Max Sequence Length | 128 |
| Training Dataset | TinyStories |

## Building

### Prerequisites

- C++17 compatible compiler (g++ or clang++)
- Python 3.x with transformers library (for tokenizer export)

### Compile

```bash
make
```

The Makefile uses the following optimization flags:

| Flag | Purpose |
|------|---------|
| -std=c++17 | C++17 standard |
| -O3 | Maximum optimization level |
| -march=native | Enable CPU-specific SIMD instructions |
| -ffast-math | Aggressive floating-point optimizations |
| -funroll-loops | Loop unrolling for performance |

### Export Tokenizer

Before running inference, export the GPT-2 tokenizer vocabulary:

```bash
python tokenizer.py
```

This creates `gpt2_vocab.json` and `merges.txt` in the project root.

## Usage

### Running Inference

```bash
./a.out
```

The Runner class initializes the model with weights from the `weights/` directory and performs autoregressive generation.

### Customizing Generation

Edit the `run()` function in `runner.hpp` to modify:

| Parameter | Default | Description |
|-----------|---------|-------------|
| prompt | "Hello, how are you?" | Input text for generation |
| max_new_tokens | 5 | Maximum tokens to generate |
| temperature | 0.8 | Sampling temperature (0 = greedy) |
| seq_len | 128 | Maximum context length |

### Example Output

```
Model config: vocab=50257, d_model=256, heads=8, blocks=6
Loaded vocab: 50257 tokens
Loaded merges: 50000 BPE merge rules
Model initialized!
Tokenized: 6 tokens
Generating...
after embeddings: 6x256
after block 0: 6x256
after block 1: 6x256
after block 2: 6x256
after block 3: 6x256
after block 4: 6x256
after block 5: 6x256
logits: 6x50257

=== Generated Text ===
Hello, how are you? Once upon a time...
======================
```

## Tensor Operations

The Tensor class implements the following operations:

| Operation | Method | Description |
|-----------|--------|-------------|
| Matrix Multiplication | operator* | Tiled matmul with 32x32 blocks |
| Addition | operator+ | Element-wise addition |
| Division | operator/ | Scalar division |
| Transpose | t() | Matrix transpose |
| Softmax | softmax() | Row-wise softmax with numerical stability |
| Layer Normalization | LayerNorm() | Normalization with learnable parameters |
| Causal Mask | mask() | Lower triangular mask for attention |
| Concatenation | concat_horizontal() | Horizontal tensor concatenation |
| Concatenation | concat_vertical() | Vertical tensor concatenation (used for KV cache) |
| Bias Addition | add_bias() | Add bias vector to each row |

## Weight Format

Weights are stored in NumPy `.npy` format and loaded at runtime. The naming convention follows:

```
transforms.{block}.{component}.{head}.{weight|bias}.npy
```

| Component | Description |
|-----------|-------------|
| q | Query projection |
| k | Key projection |
| v | Value projection |
| join | Output projection (after attention) |
| ffn.0 | FFN intermediate layer |
| ffn.3 | FFN output layer |
| norm1 | Pre-attention layer norm |
| norm2 | Pre-FFN layer norm |

## Performance Considerations

| Optimization | Impact |
|--------------|--------|
| Tiled matrix multiplication | Reduces cache misses |
| SIMD via -march=native | 2-4x speedup on modern CPUs |
| -ffast-math | Enables vectorization of floating-point ops |
| -funroll-loops | Reduces loop overhead |
| Const references | Avoids unnecessary copies |
| Reserve for vectors | Prevents reallocations |

## Error Handling

The engine includes comprehensive error handling:

- Shape mismatch detection for all tensor operations
- Try-catch blocks around attention and forward passes
- Validation of weight dimensions during initialization
- Graceful handling of missing tokenizer files
- Detailed error messages with tensor shapes

## Limitations

| Limitation | Notes |
|------------|-------|
| Single GPU | CPU-only implementation |
| Batch Size | Supports batch size of 1 |
| Precision | FP32 only (no FP16/INT8 quantization) |
| Context | Fixed maximum context length |

## Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| npy.hpp | NumPy file parsing | GitHub (external) |
| json.hpp | JSON parsing | GitHub (external) |
| transformers | Tokenizer export | HuggingFace (Python) |
| libomp | OpenMP support for parallelization | Homebrew (macOS) |

## License

This project is for educational and research purposes.

## Acknowledgments

- TinyStories dataset for model training
- HuggingFace transformers for tokenizer reference
- GPT-2 architecture as the model foundation

---

**Note:** AI assistance was used during development to locate bugs, fix bugs, add error handling, and write boilerplate code. The `npy.hpp` and `json.hpp` files are not written by the author; they are external libraries obtained from GitHub. The `tokenizer.hpp` implementation was written by Claude Sonnet 4.6.
