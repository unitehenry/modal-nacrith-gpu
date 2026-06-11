
[Nacrith-GPU](https://github.com/robtacconelli/Nacrith-GPU) compression system running on Modal serverless.

## Getting Started

```bash
# 1. Authenticate with modal
uvx modal setup

# 2. Compress file
scripts/compress path/to/file.txt

# 3. Decompress file
scripts/decompress file.txt.nc
```

## Throughput

> With a single worker on the GTX 1050 Ti, Nacrith achieves ∼50–70 tokens/second at the start of a file, settling to ∼20–30 tok/s as the KV cache fills to its 2,048-token steady state (attention cost scales linearly with cached positions). With 3 parallel workers (the maximum for 4 GB VRAM), aggregate throughput scales to ∼60–90 tok/s [^1]

### Benchmarks

| Method   | Per-Item Size | Total Size (20 items) | Notes                  |
|----------|---------------|-----------------------|------------------------|
| Original | 250 KB        | 5 MB                  | Uncompressed baseline  |
| Nacrith  | 8 KB          | 160 KB                |                        |
| Zstd     | -             | 260 KB                | Compressed from 5 MB   |

This Modal app runs inference on an H100 which gets ~500-1000 tok/s. As a benchmark, a ~5MB log file (~3M tokens) takes about an hour to compress. [^2]

### Batch Compression

There is a batch compression function that break text-based file contents into n-chunks and spawn a `compress` instance to perform compression on each chunk concurrently.

```bash
scripts/batch path/to/file.txt
```

[^1]: [Nacrith: Neural Lossless Compression via Ensemble Context Modeling and High-Precision CDF Coding](https://arxiv.org/abs/2602.19626)

[^2]: [Loghub: Apache](https://github.com/logpai/loghub)
