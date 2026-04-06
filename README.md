# Inference Engine

A custom high-performance C++ inference engine for transformer-based language models. This engine is designed to run a 30 million parameter model trained on the TinyStories dataset.

## Overview

This project implements a complete inference pipeline for decoder-only transformer models. The engine features optimized matrix multiplication with SIMD instructions, a GPT-2 compatible Byte Pair Encoding tokenizer, and a modular architecture that separates tensor operations, model components, and tokenization.

## Features

| Feature | Description |
|---------|-------------|
| SIMD Optimization | Matrix multiplication utilizes AVX2/SSE instructions via compiler flags |
| Tiled MatMul | Cache-efficient 32x32 tiling for improved memory access patterns |
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
| Threading | Single-threaded execution |

## Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| npy.hpp | NumPy file parsing | GitHub (external) |
| json.hpp | JSON parsing | GitHub (external) |
| transformers | Tokenizer export | HuggingFace (Python) |

## License

This project is for educational and research purposes.

## Acknowledgments

- TinyStories dataset for model training
- HuggingFace transformers for tokenizer reference
- GPT-2 architecture as the model foundation

---

**Note:** AI assistance was used during development to locate bugs, fix bugs, add error handling, and write boilerplate code. The `npy.hpp` and `json.hpp` files are not written by the author; they are external libraries obtained from GitHub. The `tokenizer.hpp` implementation was written by Claude Sonnet 4.6.
