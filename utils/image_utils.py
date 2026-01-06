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


def decode_image(file_bytes: bytes = None, image_bytes: bytes = None) -> Optional[np.ndarray]:
    """
    Decode image from bytes, handling EXIF orientation.
    
    Uses TurboJPEG for JPEG files (2-3x faster) and PIL for other formats.
    
    Accepts either `file_bytes` or `image_bytes` for backward compatibility.
    
    Args:
        file_bytes: Raw image bytes (preferred parameter name)
        image_bytes: Alternative name for raw image bytes
    
    Returns:
        OpenCV image (BGR) or None if decoding fails
    """
    # Normalize argument name for backward compatibility
    if file_bytes is None and image_bytes is not None:
        file_bytes = image_bytes
    if file_bytes is None:
        return None
    
    # Import TurboJPEG decoder  
    from utils.turbo_decoder import decode_image_smart
    
    # Use smart decoder (TurboJPEG for JPEG, PIL for others)
    return decode_image_smart(file_bytes)


def encode_image(img: np.ndarray, format: str = '.jpg') -> bytes:
    """
    Encode image to bytes.
    
    Args:
        img: OpenCV image (BGR format)
        format: Image format extension
    
    Returns:
        Encoded image bytes
    """
    # Convert BGR to RGB before encoding for correct color in JPEG/PNG
    # cv2.imencode doesn't auto-convert, it encodes whatever you give it
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    _, buffer = cv2.imencode(format, img_rgb)
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
