"""
Image utility functions for VowScan.
Handles image resizing, EXIF rotation, and format conversions.
"""

import cv2
import numpy as np
import base64
import io
from typing import Optional
from PIL import Image, ImageOps

from config import MAX_IMAGE_SIZE


def resize_image(img: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """
    Resize image to have longest side = max_size.
    
    Args:
        img: OpenCV image (BGR format)
        max_size: Maximum size for longest dimension
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    if max(h, w) <= max_size:
        return img
    
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_image(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode image from bytes, handling EXIF orientation.
    
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
        # Fallback to pure OpenCV if PIL fails
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except:
            return None


def encode_image(img: np.ndarray, format: str = '.jpg') -> bytes:
    """
    Encode image to bytes.
    
    Args:
        img: OpenCV image
        format: Image format extension
        
    Returns:
        Encoded image bytes
    """
    _, buffer = cv2.imencode(format, img)
    return buffer.tobytes()


def encode_image_to_base64(img: np.ndarray, format: str = '.jpg') -> str:
    """
    Encode image to base64 string (including data URI prefix).
    
    Args:
        img: OpenCV image
        format: Image format extension
        
    Returns:
        Base64 string ready for HTML img src
    """
    image_bytes = encode_image(img, format)
    b64_string = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = "image/jpeg" if format in ['.jpg', '.jpeg'] else "image/png"
    return f"data:{mime_type};base64,{b64_string}"
