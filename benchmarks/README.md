# VowImager Benchmark Tool

Comprehensive benchmarking and analysis tool for the face detection pipeline.

## Quick Start

```bash
# Standard benchmark (all images in RAM)
python benchmark.py --mode standard --images 100

# Memory-optimized (streaming batches)
python benchmark.py --mode optimized --images 1000 --stream-batch 200

# Memory usage analysis
python benchmark.py --mode analysis

# Speed/RAM tradeoff comparison
python benchmark.py --mode tradeoff
```

## Modes

### 1. Standard Benchmark (`--mode standard`)
- Loads all images into RAM at once
- Maximum speed, high RAM usage
- Best for: Testing peak performance

### 2. Optimized Benchmark (`--mode optimized`)
- Streams images in configurable batches
- Lower RAM, moderate speed
- Best for: Production deployment testing
- Configure with `--stream-batch` (default: 100)

### 3. Memory Analysis (`--mode analysis`)
- Shows RAM breakdown by component
- Identifies optimization opportunities
- No actual benchmark run

### 4. Tradeoff Comparison (`--mode tradeoff`)
- Compares different configurations
- Shows speed/RAM tradeoffs
- No actual benchmark run

## Options

- `--images N` - Number of images to process (default: 1000)
- `--stream-batch N` - Batch size for streaming mode (default: 100)
- `--warmup` - Run warmup before benchmark

## Examples

```bash
# Quick test with 100 images
python benchmark.py --images 100 --warmup

# Production test: 200 image batches (recommended)
python benchmark.py --mode optimized --images 1000 --stream-batch 200

# Compare configurations
python benchmark.py --mode tradeoff
```

## Results Files

Benchmark results are saved to `benchmark_results.json` (not auto-saved in current version).

## Cleanup

Old benchmark files have been consolidated. Previous results are in:
- `benchmark_results.json`
- `benchmark_results_memory_optimized.json`
- `benchmark_results_with_monitoring.json`
