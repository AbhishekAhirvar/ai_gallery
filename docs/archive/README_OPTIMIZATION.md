# VowImager - GPU-Accelerated Face Recognition

## System Requirements

### Python Dependencies
```bash
pip install PyTurboJPEG exifread
```

### System Dependencies (for TurboJPEG)

**Ubuntu/Debian:**
```bash
sudo apt-get install libturbojpeg0-dev
```

**macOS:**
```bash
brew install jpeg-turbo
```

**Note**: If libturbojpeg is not available, the system will automatically fall back to PIL (slower but functional).

## Quick Start

```bash
cd /home/abhishekverma/Projects/vow/VowImager
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Performance Testing

### Run Benchmark
```bash
python benchmark.py --images /path/to/test/images --num-images 100
```

### Monitor GPU
```bash
# In separate terminal
watch -n 0.5 nvidia-smi
```

### Expected Output (Console)
```
📊 Batch 1: 32 images → 85 faces in 2.35s (13.6 img/s)
...
============================================================
📈 PERFORMANCE REPORT
============================================================
Total Images: 100
🚀 THROUGHPUT: 35.2 images/second
📊 Est. time for 1000 images: 28.4s
============================================================
```

## Configuration

Edit `config.py`:
```python
PREPROCESSING_WORKERS = 4      # CPU cores (4-8 recommended)
GPU_BATCH_SIZE = 32           # GPU batch size
MAX_IMAGE_SIZE = 640          # Detection resolution
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| TurboJPEG not available | Install `libturbojpeg0-dev` (Linux) or `jpeg-turbo` (macOS) |
| GPU not detected | Check `nvidia-smi` and `onnxruntime-gpu` installation |
| Out of memory | Reduce `GPU_BATCH_SIZE` to 16-24 |
| Slow performance | Check console logs for bottleneck breakdown |

## Performance Targets

| Scenario | Target | Status |
|----------|--------|--------|
| 100 images | <5 seconds | ✅ Expected |
| 500 images | <15 seconds | ✅ Expected |
| 1000 images | <30 seconds | ✅ Expected |

Phase 1+2 should achieve **30-50 images/second** on GTX 1650.
