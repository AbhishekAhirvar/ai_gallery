"""
Face Detection Service for VowScan.
Wraps InsightFace for high-quality face detection and embedding extraction.
Uses 'buffalo_l' model pack.
"""

import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from dataclasses import dataclass
from typing import List
import streamlit as st
from config import DET_CONF_THRESH

@dataclass
class FaceEmbedding:
    """Represents a face embedding with metadata."""
    embedding: np.ndarray
    filename: str
    face_index: int = 0
    confidence: float = 0.0
    facial_area: dict = None  # x, y, w, h
    landmarks: np.ndarray = None # 5 keypoints
    thumbnail_path: str = None


@st.cache_resource
def get_face_analyser():
    """
    Initialize and cache the InsightFace model.
    """
    # Initialize InsightFace with buffalo_l model
    app = FaceAnalysis(name='buffalo_l')
    
    # Auto-detect context: 0 for GPU, -1 for CPU
    # We try 0 first if ONNX Runtime GPU is available
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        app.prepare(ctx_id=-1, det_size=(640, 640))
        
    return app


class FaceDetector:
    """
    Service for detecting faces and extracting embeddings using InsightFace.
    """
    
    def __init__(self):
        self.app = get_face_analyser()
        self._thumbnails_dir = "thumbnails"
        if not os.path.exists(self._thumbnails_dir):
            os.makedirs(self._thumbnails_dir)
    
    def extract_embeddings(
        self,
        img: np.ndarray,
        filename: str
    ) -> List[FaceEmbedding]:
        """
        Extract face embeddings from an image.
        
        Args:
            img: OpenCV image (BGR format)
            filename: Original filename for reference
            
        Returns:
            List of FaceEmbedding objects
        """
        try:
            # InsightFace expects BGR image (OpenCV format)
            faces = self.app.get(img)
            
            embeddings_list = []
            
            for idx, face in enumerate(faces):
                # InsightFace Face object attributes:
                # bbox: [x1, y1, x2, y2]
                # embedding: (512,)
                # det_score: float
                
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Quality Gate
                if face.det_score < DET_CONF_THRESH:
                    continue
                
                w_box = x2 - x1
                h_box = y2 - y1
                
                if w_box <= 0 or h_box <= 0:
                    continue
                
                # Thumbnail generation with padding (Squarish for better avatar look)
                h_img, w_img, _ = img.shape
                
                # Calculate center and max dimension
                cx = x1 + w_box // 2
                cy = y1 + h_box // 2
                max_dim = max(w_box, h_box)
                
                # Add 10% padding for tight face focus
                context_size = int(max_dim * 1.1)
                half_size = context_size // 2
                
                tx1 = max(0, cx - half_size)
                ty1 = max(0, cy - half_size)
                tx2 = min(w_img, cx + half_size)
                ty2 = min(h_img, cy + half_size)
                
                face_crop = img[ty1:ty2, tx1:tx2]
                
                thumb_path = None
                if face_crop.size > 0:
                    safe_filename = filename.replace('.', '_')
                    thumb_filename = f"thumb_{safe_filename}_{idx}.jpg"
                    thumb_path = os.path.join(self._thumbnails_dir, thumb_filename)
                    cv2.imwrite(thumb_path, face_crop)
                
                embeddings_list.append(
                    FaceEmbedding(
                        embedding=face.embedding,
                        filename=filename,
                        face_index=idx,
                        confidence=float(face.det_score),
                        facial_area={'x': int(x1), 'y': int(y1), 'w': int(w_box), 'h': int(h_box)},
                        landmarks=face.kps,
                        thumbnail_path=thumb_path
                    )
                )
                
            return embeddings_list
            
        except Exception as e:
            # print(f"Processing error: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources if needed."""
        # Models are cached, nothing specific to clean
        pass
