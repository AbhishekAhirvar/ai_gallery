"""
VowScan Configuration
All configurable constants in one place for easy tuning.
"""

# InsightFace Configuration
# Models are downloaded automatically by InsightFace


# Image Processing
MAX_IMAGE_SIZE = 1280  # Longest side in pixels

# DBSCAN Clustering
DBSCAN_EPS = 0.7  # Increased to merge duplicates better
DBSCAN_MIN_SAMPLES = 3  # Minimum 3 faces to form a person group

# Detection Quality
DET_CONF_THRESH = 0.50


# UI Configuration
GALLERY_COLUMNS = 4
DEFAULT_EXPANDED_GROUPS = 3

# Supported file formats
# Supported file formats
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
SUPPORTED_UPLOAD_TYPES = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'zip']

MAX_RAM_IMAGES = 100
MAX_RAM_SIZE_MB = 1000  #