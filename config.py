"""
VowScan Configuration
All configurable constants in one place for easy tuning.
"""

# DeepFace Configuration
FACE_MODEL = "Facenet512"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = True

# Image Processing
MAX_IMAGE_SIZE = 640  # Longest side in pixels

# DBSCAN Clustering
DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 2

# UI Configuration
GALLERY_COLUMNS = 4
DEFAULT_EXPANDED_GROUPS = 3

# Supported file formats
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
SUPPORTED_UPLOAD_TYPES = ['jpg', 'jpeg', 'png', 'zip']
