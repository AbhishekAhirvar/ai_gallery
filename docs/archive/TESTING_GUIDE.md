# Phase 1 & 2 Optimization - Testing Guide

## What Was Implemented

### Phase 1: TurboJPEG Fast Decoding
- **Created**: `utils/turbo_decoder.py` with hardware-accelerated JPEG decoder
- **Modified**: `utils/image_utils.py` to use TurboJPEG for JPEG files (2-3x faster)
- **Added**: EXIF orientation handling for rotated images
- **Fallback**: Automatic fallback to PIL for non-JPEG formats

### Phase 2: Parallel Preprocessing
- **Modified**: `app.py` with ThreadPoolExecutor for concurrent image preprocessing
- **Workers**: 4 parallel CPU threads for decode/resize operations
- **Batch Processing**: Maintains GPU batch inference (32 images per batch)
- **Performance Tracking**: Added detailed timing logs

## Performance Monitoring

### During Processing (Console Output)

When you run the app, you'll see detailed logs in your terminal:

```
📊 Batch 1: 32 images → 85 faces in 2.35s (13.6 img/s)
📊 Batch 2: 32 images → 78 faces in 2.21s (14.5 img/s)
...
============================================================
📈 PERFORMANCE REPORT
============================================================
Total Images: 100
Total Faces Detected: 342
Person Clusters: 12

⏱️  TIMING BREAKDOWN:
  Decode + Resize:  3.45s (28.7%) - 29.0 img/s
  Face Detection:   8.12s (67.5%) - 12.3 img/s
  Clustering:       0.46s (3.8%)
  TOTAL:            12.03s

🚀 THROUGHPUT: 8.31 images/second
📊 Est. time for 1000 images: 120.3s
============================================================
```

### In Streamlit UI

After processing completes, you'll see:
- ✅ Success message with total time and throughput
- 📊 Statistics showing faces detected and clusters formed

## How to Test

### Method 1: Run the Streamlit App

```bash
cd /home/abhishekverma/Projects/vow/VowImager
source venv/bin/activate
streamlit run app.py
```

1. Upload wedding photos (ZIP file or individual images)
2. Processing starts automatically
3. Watch the terminal/console for detailed performance logs
4. Check the final performance summary

### Method 2: Run Benchmark Script

For more controlled testing:

```bash
cd /home/abhishekverma/Projects/vow/VowImager
source venv/bin/activate

# Test with your own images
python benchmark.py --images /path/to/test/images --num-images 100 --mode both

# Or test with images in originals/ folder
python benchmark.py --mode both
```

The benchmark will output:
- Decode performance (TurboJPEG vs PIL comparison)
- Full pipeline performance (decode + detection + clustering)
- JSON file with detailed results (optional: `--output results.json`)

## Expected Performance Improvements

| Metric | Before (Baseline) | After Phase 1+2 | Speedup |
|--------|-------------------|-----------------|---------|
| Decode Speed | ~100ms/image | ~30-50ms/image | 2-3x |
| CPU Utilization | Serial (1 core) | Parallel (4 cores) | 2-3x |
| Overall Throughput | 10-15 img/s | 30-50 img/s | 3-4x |
| 1000 Images | 60-90 seconds | 20-30 seconds | 3-4x |

## Troubleshooting

### TurboJPEG Not Available

If you see: `⚠️ TurboJPEG not available: ...`

**On Ubuntu/Debian:**
```bash
sudo apt-get install libturbojpeg0-dev
pip install PyTurboJPEG
```

**On macOS:**
```bash
brew install jpeg-turbo
pip install PyTurboJPEG
```

The system will automatically fall back to PIL if TurboJPEG is unavailable, but performance will be slower.

### GPU Not Being Used

If you see: `Using CPU for inference`

Check:
```bash
nvidia-smi  # Should show your GPU
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```

## Performance Tuning

### Configuration Options (config.py)

```python
# Adjust these for your hardware:
PREPROCESSING_WORKERS = 4      # CPU cores for parallel decode (default: 4)
GPU_BATCH_SIZE = 32           # GPU batch size (default: 32)
MAX_IMAGE_SIZE = 640          # Image resize target (default: 640)
```

**Recommendations:**
- **More CPU cores**: Increase `PREPROCESSING_WORKERS` to 6-8
- **More GPU memory**: Increase `GPU_BATCH_SIZE` to 48-64
- **Less GPU memory**: Decrease `GPU_BATCH_SIZE` to 16-24
- **Faster processing**: Decrease `MAX_IMAGE_SIZE` to 512 (slight accuracy loss)

## Monitoring GPU Utilization

In a separate terminal, run:
```bash
watch -n 0.5 nvidia-smi
```

**What to look for:**
- GPU utilization should spike to >80% during face detection batches
- Memory usage should stay below 3.5GB on GTX 1650 (4GB total)

## Next Steps

After testing Phase 1 & 2:

1. **If target achieved** (<20s for 1000 images): You're done! 🎉
2. **If still slow**: Consider Phase 3 (NVIDIA DALI) for GPU-accelerated decode
3. **If good enough**: Stop here to avoid added complexity

Let me know your performance results and we can decide whether to proceed with Phase 3 (DALI) or Phase 4 (FAISS GPU clustering).
