"""
TurboJPEG Fast Image Decoder for VowScan.
Provides 2-3x faster JPEG decoding compared to PIL.
"""

import io
import cv2
import numpy as np
import exifread
from pathlib import Path
from typing import Optional, Union
from PIL import Image, ImageOps

# Try to import TurboJPEG, fall back gracefully if not available
try:
    from turbojpeg import TurboJPEG
    try:
        _turbo_jpeg = TurboJPEG()
        TURBOJPEG_AVAILABLE = True
        print("✅ TurboJPEG library loaded successfully")
    except Exception as init_error:
        # TurboJPEG module exists but library not found
        TURBOJPEG_AVAILABLE = False
        _turbo_jpeg = None
        print(f"⚠️ TurboJPEG library not found: {init_error}")
        print("   Falling back to PIL (slower). To install:")
        print("   Ubuntu/Debian: sudo apt-get install libturbojpeg0-dev")
        print("   macOS: brew install jpeg-turbo")
except ImportError as e:
    TURBOJPEG_AVAILABLE = False
    _turbo_jpeg = None
    print(f"⚠️ TurboJPEG module not installed: {e}")
    print("   Install with: pip install PyTurboJPEG")


def _get_exif_orientation(file_bytes: bytes) -> int:
    """
    Extract EXIF orientation tag from image bytes.
    
    Args:
        file_bytes: Raw image bytes
        
    Returns:
        EXIF orientation value (1-8), defaults to 1 (no rotation)
    """
    try:
        tags = exifread.process_file(io.BytesIO(file_bytes), details=False, stop_tag='Image Orientation')
        orientation_tag = tags.get('Image Orientation')
        
        if orientation_tag:
            # ExifRead returns IfdTag, need to convert to int
            return int(str(orientation_tag.values[0]))
        return 1  # Default: no rotation
    except Exception:
        return 1


def _apply_exif_rotation(img: np.ndarray, orientation: int) -> np.ndarray:
    """
    Apply EXIF orientation transformation to image.
    
    Args:
        img: OpenCV image (BGR format)
        orientation: EXIF orientation value (1-8)
        
    Returns:
        Rotated image
    """
    if orientation == 1:
        return img  # No rotation needed
    elif orientation == 2:
        return cv2.flip(img, 1)  # Flip horizontal
    elif orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 4:
        return cv2.flip(img, 0)  # Flip vertical
    elif orientation == 5:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.flip(img, 1)
    elif orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return cv2.flip(img, 1)
    elif orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return img


def decode_image_turbo(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode JPEG image using TurboJPEG with EXIF orientation handling.
    
    This is 2-3x faster than PIL for JPEG images.
    
    Args:
        file_bytes: Raw image bytes
        
    Returns:
        OpenCV image (BGR) or None if decoding fails
    """
    if not TURBOJPEG_AVAILABLE or _turbo_jpeg is None:
        return None
    
    try:
        # Decode JPEG to RGB using TurboJPEG (hardware accelerated)
        img_rgb = _turbo_jpeg.decode(file_bytes)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Apply EXIF rotation if needed
        orientation = _get_exif_orientation(file_bytes)
        if orientation != 1:
            img_bgr = _apply_exif_rotation(img_bgr, orientation)
        
        return img_bgr
    except Exception:
        # TurboJPEG failed, return None to trigger fallback
        return None


def decode_image_pil(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode image using PIL (slower but supports all formats).
    
    Args:
        file_bytes: Raw image bytes
        
    Returns:
        OpenCV image (BGR) or None if decoding fails
    """
    try:
        # Use PIL to handle EXIF rotation automatically
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        
        # Convert to OpenCV format (RGB -> BGR)
        img_np = np.array(image)
        
        if len(img_np.shape) == 2:  # Grayscale
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    except Exception:
        # Final fallback to pure OpenCV
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


def is_jpeg(file_bytes: bytes, filename: Optional[str] = None) -> bool:
    """
    Check if image is JPEG format.
    
    Args:
        file_bytes: Raw image bytes
        filename: Optional filename for extension check
        
    Returns:
        True if JPEG format
    """
    # Check by extension first (fast)
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            return True
    
    # Check magic bytes (JPEG starts with FFD8)
    if len(file_bytes) >= 2:
        return file_bytes[0:2] == b'\xff\xd8'
    
    return False


def decode_image_smart(file_bytes: bytes, filename: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Smart image decoder that uses TurboJPEG for JPEGs and PIL for other formats.
    
    This function automatically selects the fastest decoder:
    - TurboJPEG for JPEG files (2-3x faster)
    - PIL for PNG, BMP, TIFF, etc.
    
    Args:
        file_bytes: Raw image bytes
        filename: Optional filename for format detection
        
    Returns:
        OpenCV image (BGR) or None if decoding fails
    """
    # Try TurboJPEG for JPEG files
    if TURBOJPEG_AVAILABLE and is_jpeg(file_bytes, filename):
        img = decode_image_turbo(file_bytes)
        if img is not None:
            return img
    
    # Fall back to PIL for non-JPEG or if TurboJPEG failed
    return decode_image_pil(file_bytes)


# Benchmark function for testing
def benchmark_decoder(image_path: str, iterations: int = 100) -> dict:
    """
    Benchmark TurboJPEG vs PIL decoding speed.
    
    Args:
        image_path: Path to test image
        iterations: Number of iterations to run
        
    Returns:
        Dict with timing results
    """
    import time
    from pathlib import Path
    
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    
    filename = Path(image_path).name
    
    results = {
        'image_path': image_path,
        'file_size_mb': len(file_bytes) / (1024 * 1024),
        'iterations': iterations,
        'is_jpeg': is_jpeg(file_bytes, filename)
    }
    
    # Benchmark TurboJPEG
    if TURBOJPEG_AVAILABLE and is_jpeg(file_bytes, filename):
        start = time.time()
        for _ in range(iterations):
            img = decode_image_turbo(file_bytes)
        turbo_time = time.time() - start
        results['turbojpeg_time_ms'] = (turbo_time / iterations) * 1000
        results['turbojpeg_available'] = True
    else:
        results['turbojpeg_time_ms'] = None
        results['turbojpeg_available'] = False
    
    # Benchmark PIL
    start = time.time()
    for _ in range(iterations):
        img = decode_image_pil(file_bytes)
    pil_time = time.time() - start
    results['pil_time_ms'] = (pil_time / iterations) * 1000
    
    # Calculate speedup
    if results['turbojpeg_time_ms']:
        results['speedup'] = results['pil_time_ms'] / results['turbojpeg_time_ms']
    else:
        results['speedup'] = 1.0
    
    return results
