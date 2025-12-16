"""
Image utility functions for VowScan.
Handles image resizing and format conversions.
"""

import cv2
import numpy as np
from typing import Optional

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
    Decode image from bytes.
    
    Args:
        file_bytes: Raw image bytes
        
    Returns:
        OpenCV image or None if decoding fails
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


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
