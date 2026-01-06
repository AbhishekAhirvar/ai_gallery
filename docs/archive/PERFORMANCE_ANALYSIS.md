# VowImager Performance Analysis

## Latest Benchmark: 1000 Images (January 6, 2026)

### Performance Summary

| Metric | Value |
|--------|-------|
| **Images Processed** | 1000 |
| **Faces Detected** | 1790 |
| **Processing Time** | 30.021 seconds |
| **Throughput** | **33.31 images/second** |
| **Clusters** | 190 |

### Resource Utilization

#### CPU
- **Average**: 29.6% (Global), 116.0% (Process)
- **Peak**: 64.4% (Global), 237.2% (Process)

#### RAM
- **Process Peak**: 8.71 GB
- **System Peak**: 20.83 GB (66.6% of total)

#### GPU (NVIDIA GeForce GTX 1650)
- **Utilization**: 76.7% average, 98.0% peak
- **VRAM**: 3.0 GB average, 3.07 GB peak (74.9% of 4GB)
- **Temperature**: 42.7°C average, 47.0°C peak

### Processing Breakdown
- **Detection & Recognition**: 28.774s (95.8%)
  - Detection: 19.721s (68.5%)
  - Recognition: 8.228s (28.6%)
- **Clustering**: 1.247s (4.2%)

---

# VowImager Performance Analysis (Historical) & Optimization Guide

## Current Performance

### Before Optimizations
- **Speed**: 2-3 images/second
- **Image Size**: ~4MB per image
- **GPU**: GTX 1650 (CUDA enabled)
- **Model**: buffalo_l (ResNet50 backbone)

### After Quick Fixes (✅ IMPLEMENTED)
- **Expected Speed**: 10-15 images/second
- **Optimizations Applied**:
  1. ✅ Removed thumbnail generation during detection
  2. ✅ Removed disk I/O (medium/thumb storage)
  3. ✅ Increased BATCH_SIZE from 16 → 32
  4. ✅ Session state caching (prevents reprocessing)
  5. ✅ MAX_IMAGE_SIZE reduced to 640px

---

## Current Architecture Pipeline

### Processing Flow (Per Image)
```
1. File Upload (Streamlit) → Memory
   ↓
2. extract_images_from_uploads() → Disk (originals/)
   ↓
3. Read from disk → bytes
   ↓
4. decode_image() 
   - PIL.Image.open() ← SLOW (EXIF rotation)
   - ImageOps.exif_transpose() ← CPU-bound
   - Convert RGB→BGR
   ↓
5. resize_image(img, 640) ← SLOW
   - cv2.resize() on large image
   ↓
6. encode_image() to bytes ← SLOW
   - cv2.imencode()
   ↓
7. resize_image(img, 300) for thumbnail ← SLOW
   - Another resize operation
   ↓
8. encode_image() to bytes ← SLOW
   - Another encode
   ↓
9. store_derived_images() ← DISK I/O (SLOW)
   - Write medium.jpg to disk
   - Write thumb.jpg to disk
   ↓
10. Accumulate in batch (16 images)
   ↓
11. extract_embeddings(batch) → GPU ← FAST
   - Face detection
   - Feature extraction
```

### Time Breakdown (Estimated per 4MB image)

#### BEFORE Optimization:
```
1. File read from disk:        ~10ms
2. PIL decode + EXIF:          ~100ms  ← BOTTLENECK #1
3. Resize to 640px:            ~50ms   
4. Encode to bytes:            ~40ms   ← REMOVED
5. Resize to 300px:            ~20ms   ← REMOVED
6. Encode to bytes:            ~30ms   ← REMOVED
7. Disk write (2 files):       ~60ms   ← REMOVED
8. GPU face detection:         ~30ms   (batched, so ~2ms per image)
   
TOTAL: ~340ms per image = ~3 images/second
```

#### AFTER Optimization:
```
1. File read from disk:        ~10ms
2. PIL decode + EXIF:          ~100ms  ← Still main bottleneck
3. Resize to 640px:            ~50ms   
4. GPU face detection:         ~30ms   (batched BATCH_SIZE=32, so ~1ms per image)
   
TOTAL: ~161ms per image = ~6 images/second (minimum)
With batching effects: ~80-100ms per image = 10-12 images/second
```

**Speedup: 3.3-4x faster** ✅

---

## Current Config Values

### `config.py`
```python
MAX_IMAGE_SIZE = 640          # Image resize target
DET_CONF_THRESH = 0.50        # Face detection confidence
DBSCAN_EPS = 0.7              # Clustering threshold
DBSCAN_MIN_SAMPLES = 3        # Min faces per cluster
```

### `app.py`
```python
BATCH_SIZE = 16               # Images processed together
```

### `face_detector.py`
```python
MODEL = 'buffalo_l'           # ResNet50-based (accurate but slow)
DET_SIZE = (640, 640)         # Detection resolution
```

---

## BOTTLENECK ANALYSIS

### 🔴 PRIMARY BOTTLENECKS (90% of time)

#### 1. **PIL EXIF Rotation** (~100ms/image)
- **Location**: `utils/image_utils.py:60-62`
- **Issue**: PIL.Image.open() + exif_transpose() is CPU-intensive
- **Impact**: 30% of total time
- **Fix Options**:
  - Use `cv2.imdecode()` directly (4x faster, but lose EXIF)
  - Use `turbojpeg` library (2x faster, keeps EXIF)
  - Skip EXIF for non-rotated images

#### 2. **Double Resize/Encode** (~140ms/image)
- **Location**: `app.py:119-128`
- **Issue**: Creating medium (640px) AND thumb (300px) during detection
- **Impact**: 40% of total time
- **Fix**: Skip derived image creation during detection, generate on-demand

#### 3. **Disk I/O** (~60ms/image)
- **Location**: `app.py:128` - `store_derived_images()`
- **Issue**: Writing medium.jpg and thumb.jpg every image
- **Impact**: 18% of total time
- **Fix**: Skip storage during detection, generate thumbnails only for final results

#### 4. **Serial Processing** 
- **Issue**: Steps 1-9 happen serially (one image at a time)
- **GPU Utilization**: Only ~10% of pipeline time
- **Fix**: Parallelize CPU preprocessing

---

## OPTIMIZATION OPPORTUNITIES

### 🚀 HIGH IMPACT (Easy Wins)

#### Option 1: Skip Derived Images During Detection ⭐⭐⭐⭐⭐
**Expected speedup: 3-4x (from 3 img/s → 10-12 img/s)**

```python
# In app.py, remove lines 119-128
# Don't create medium/thumb during detection
# Just decode → resize to 640 → detect
# Generate derived images later when displaying results
```

**Changes needed**:
- Comment out `store_derived_images()` in processing loop
- Generate thumbnails in gallery rendering (from original or on-demand)

#### Option 2: Use TurboJPEG for Decoding ⭐⭐⭐⭐
**Expected speedup: 2x on decode (100ms → 50ms)**

```bash
pip install PyTurboJPEG
```

```python
# In image_utils.py
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

def decode_image_fast(file_bytes):
    img = jpeg.decode(file_bytes)  # RGB format
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```

#### Option 3: Increase Batch Size ⭐⭐⭐
**Expected speedup: 1.2x**

```python
# In app.py
BATCH_SIZE = 32  # or even 64 if you have 4GB+ RAM
```

### 🔥 MEDIUM IMPACT

#### Option 4: Multiprocessing for CPU Work ⭐⭐⭐⭐
**Expected speedup: 2-3x (if you have 4+ CPU cores)**

```python
from multiprocessing import Pool

def preprocess_batch(files_batch):
    # Decode + resize in parallel
    with Pool(processes=4) as pool:
        images = pool.map(decode_and_resize, files_batch)
    return images
```

#### Option 5: Skip EXIF Rotation ⭐⭐
**Expected speedup: 1.3x**

```python
# Use cv2.imdecode directly (no PIL)
# Tradeoff: Images from phones might appear rotated
```

---

## RECOMMENDED OPTIMIZATION SEQUENCE

### Phase 1: Quick Wins ✅ COMPLETED
1. ✅ **Skipped derived image storage** during detection
2. ✅ **Increased BATCH_SIZE to 32**
3. ✅ **Result: Expected 10-15 images/second** (TEST NOW!)

### Phase 2: Library Upgrade (30 minutes) - NEXT
1. Install TurboJPEG
2. Replace PIL decoder
3. **Expected result: 20-30 images/second**

### Phase 3: Parallelization (2 hours) - ADVANCED
1. Implement multiprocessing for decode/resize
2. Async I/O for file reading
3. **Expected result: 40-60 images/second**

---

## CURRENT SYSTEM SPECS

### Hardware
```
GPU: NVIDIA GTX 1650 (4GB VRAM)
CUDA: 13.0
cuDNN: 9.x
```

### Software
```
Python: 3.12
onnxruntime-gpu: 1.20.1
insightface: (check with pip show insightface)
opencv-python: (check version)
Streamlit: (check version)
```

---

## PROFILING COMMANDS

To measure actual bottlenecks:

```python
# Add to app.py
import time

start = time.time()
# ... operation ...
print(f"Step took: {time.time() - start:.3f}s")
```

Or use `cProfile`:
```bash
python -m cProfile -o profile.stats app.py
python -m pstats profile.stats
```

---

## ALTERNATIVE ARCHITECTURES

### Option A: Streaming Pipeline
```
Upload → Queue → [Worker 1: Decode] → Queue → [Worker 2: Resize] → Queue → [Worker 3: GPU Batch] → Results
```

### Option B: GPU-Accelerated Preprocessing
```python
# Use NVIDIA DALI or cv2.cuda for GPU resize/decode
import cv2.cuda
# Decode and resize on GPU before face detection
```

### Option C: Lazy Loading
```
Don't process all images upfront
Process on-demand when user views gallery
Cache results in session state
```

---

## KEY METRICS TO TRACK

1. **Images/second** (overall throughput)
2. **GPU utilization** (should be >80% during detection)
3. **Memory usage** (watch for OOM with larger batches)
4. **Time per pipeline stage** (identify bottlenecks)

---

## QUICK TEST SCRIPT

Save as `benchmark.py`:

```python
import time
import cv2
import numpy as np
from services.face_detector import FaceDetector
from utils.image_utils import decode_image, resize_image

# Load test image
with open("test_image.jpg", "rb") as f:
    img_bytes = f.read()

detector = FaceDetector()

# Benchmark
n = 100
start = time.time()

for i in range(n):
    img = decode_image(img_bytes)
    img = resize_image(img, 640)
    
total = time.time() - start
print(f"Decode+Resize: {total/n*1000:.1f}ms per image")
print(f"Throughput: {n/total:.1f} images/second")
```

---

## NEXT STEPS

1. **Measure first**: Add timing logs to confirm bottlenecks
2. **Quick win**: Remove `store_derived_images()` call
3. **Research**: TurboJPEG vs cv2.imdecode vs PIL benchmarks
4. **Consider**: Lazy loading architecture for large datasets
