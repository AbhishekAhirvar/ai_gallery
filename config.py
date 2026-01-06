"""
VowImager Configuration
Centralized configuration for the face recognition pipeline.
"""

# ==============================================================================
# 1. SYSTEM & HARDWARE
# ==============================================================================
# Batch Processing
# How many images to load/resize in parallel (CPU -> Memory)
# Higher = better CPU utilization, more RAM. 64 saturates 4 cores nicely.
IMAGE_BATCH_SIZE = 64  

# How many faces to recognize in parallel (GPU Memory Bound)
# 32 is optimal for GTX 1650 (4GB VRAM) - keep unchanged for good GPU utilization
RECOGNITION_BATCH_SIZE = 32
DEFAULT_BATCH_SIZE = 32
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 128
FACE_MEMORY_ESTIMATE_MB = 50 

# Processed Image Cleanup
DEFAULT_THUMBNAIL_MAX_AGE_HOURS = 24

# Parallelism (4-core optimized)
PREPROCESSING_WORKERS = 4  # Match CPU cores for max parallelism
USE_TURBO_JPEG = True      # Enable libjpeg-turbo acceleration

# Memory Safety (Optimized for 4GB RAM - use ~3.5GB, leave 0.5GB for OS)
MAX_RAM_IMAGES = 200       # Allow more images in RAM for faster throughput
MAX_RAM_SIZE_MB = 3500     # Use up to 3.5GB RAM for image processing

# ==============================================================================
# 2. FILE HANDLING
# ==============================================================================
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
SUPPORTED_UPLOAD_TYPES = ['zip'] + [ext.strip('.') for ext in SUPPORTED_IMAGE_EXTENSIONS]

# ==============================================================================
# 3. FACE DETECTION (InsightFace SCRFD)
# ==============================================================================
# Input Resolution
# (640, 640) offers the best trade-off for accuracy vs speed.
# Smaller sizes (e.g. 480x480) significantly degrade recall for small faces.
DET_SIZE = (640, 640)

# Confidence Threshold
# Filter out low-quality/false detections.
DET_CONF_THRESH = 0.80  # Optimized for high precision

# ==============================================================================
# 4. FACE RECOGNITION (InsightFace ArcFace)
# ==============================================================================
FACE_ALIGNMENT_SIZE = 112  # Standard input for ArcFace (112x112)
FACE_EMBEDDING_DIM = 512   # Dimension of output vectors

# ==============================================================================
# 5. CLUSTERING (HDBSCAN)
# ==============================================================================
USE_FAISS_CLUSTERING = False  # Set False to use pure HDBSCAN (Simpler, Robust)
FAISS_K_NEIGHBORS = 10        # Only returned if FAISS used

# HDBSCAN Parameters
MIN_CLUSTER_SIZE = 2      # Minimum faces to form a "person"
MIN_SAMPLES = 1           # Conservative noise handling
CLUSTERING_METRIC = 'euclidean' # 'euclidean' on normalized vectors == cosine

# ==============================================================================
# 6. FACE QUALITY & THUMBNAILS
# ==============================================================================
THUMBNAIL_SIZE = (150, 150)
THUMBNAIL_CONTEXT_MULTIPLIER = 1.1  # Crop slightly larger than face box

# Quality Scoring (for selecting best thumbnail)
THUMBNAIL_SCAN_LIMIT = 50       # Max faces to check per cluster
FACE_SIZE_QUALITY_THRESH = 15000 # px² (e.g. 122x122)
YAW_FRONTALITY_THRESH = 0.8     # >0.8 is "frontal"
CONFIDENCE_QUALITY_THRESH = 0.7 # High confidence detection

# ==============================================================================
# 7. UI CONFIGURATION
# ==============================================================================
GALLERY_COLUMNS = 4
DEFAULT_EXPANDED_GROUPS = 3
MAX_IMAGE_SIZE = 640  # Max display size in gallery