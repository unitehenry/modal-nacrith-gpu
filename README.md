
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

This Modal app runs inference on an H100 which peeks ~1000 tok/s. As a benchmark, a ~10MB log file (~3M tokens) which takes about 50 minutes to compress.

### Batch Compression

There is a batch compression function that break text-based file contents into n-chunks and spawn a `compress` instance to perform compression on each chunk concurrently.

```bash
scripts/batch path/to/file.txt
```

## References

[^1]: [Nacrith: Neural Lossless Compression via Ensemble Context Modeling and High-Precision CDF Coding](https://arxiv.org/abs/2602.19626)
