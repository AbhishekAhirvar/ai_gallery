"""
Face Detection Service for VowScan.
Wraps DeepFace for face detection and embedding extraction.
"""

import os
import tempfile
from typing import List, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from deepface import DeepFace

from config import FACE_MODEL, DETECTOR_BACKEND, ENFORCE_DETECTION


@dataclass
class FaceEmbedding:
    """Represents a face embedding with metadata."""
    embedding: np.ndarray
    filename: str
    face_index: int = 0  # For images with multiple faces


class FaceDetector:
    """
    Service for detecting faces and extracting embeddings.
    Uses DeepFace with configurable model and detector backend.
    """
    
    def __init__(
        self,
        model_name: str = FACE_MODEL,
        detector_backend: str = DETECTOR_BACKEND
    ):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self._temp_dir = None
    
    def _get_temp_dir(self) -> str:
        """Get or create temporary directory for processing."""
        if self._temp_dir is None or not os.path.exists(self._temp_dir):
            self._temp_dir = tempfile.mkdtemp(prefix="vowscan_face_")
        return self._temp_dir
    
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
            List of FaceEmbedding objects (empty if no faces found)
        """
        # Save image temporarily for DeepFace
        temp_path = os.path.join(self._get_temp_dir(), "temp_face.jpg")
        cv2.imwrite(temp_path, img)
        
        try:
            embeddings = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=ENFORCE_DETECTION
            )
            
            return [
                FaceEmbedding(
                    embedding=np.array(emb['embedding']),
                    filename=filename,
                    face_index=idx
                )
                for idx, emb in enumerate(embeddings)
            ]
            
        except Exception:
            # No face detected or other error
            return []
        
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def cleanup(self):
        """Clean up temporary resources."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
