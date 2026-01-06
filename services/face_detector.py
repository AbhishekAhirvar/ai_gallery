"""
Face Detection Service for VowScan.
Wraps InsightFace for high-quality face detection and embedding extraction.
Uses 'buffalo_l' model pack.
"""

import os
import cv2
import numpy as np
import time
import logging
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import streamlit as st
from config import (
    DET_CONF_THRESH,
    DET_SIZE,
    FACE_ALIGNMENT_SIZE,
    THUMBNAIL_CONTEXT_MULTIPLIER,
    RECOGNITION_BATCH_SIZE,
    FACE_EMBEDDING_DIM,
    DEFAULT_BATCH_SIZE,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    FACE_MEMORY_ESTIMATE_MB,
    DEFAULT_THUMBNAIL_MAX_AGE_HOURS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@st.cache_resource(ttl=3600)  # 1 hour cache timeout
def get_face_analyser():
    """
    Initialize and cache the InsightFace model.
    """
    try:
        # Initialize InsightFace with buffalo_l model
        app = FaceAnalysis(name='buffalo_l')
        
        # Auto-detect context: 0 for GPU, -1 for CPU
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        
        # Check for CUDA/ROCm (generic GPU check)
        has_gpu = any(p in providers for p in ['CUDAExecutionProvider', 'ROCMExecutionProvider', 'CoreMLExecutionProvider'])
        
        if has_gpu:
            try:
                # ctx_id=0 usually maps to the first GPU
                app.prepare(ctx_id=0, det_size=DET_SIZE)
                logger.info("✅ GPU Acceleration Enabled (InsightFace)")
            except Exception as e:
                logger.warning(f"⚠️ GPU init failed despite provider presence: {e}. Fallback to CPU.")
                app.prepare(ctx_id=-1, det_size=DET_SIZE)
        else:
            app.prepare(ctx_id=-1, det_size=DET_SIZE)
            logger.info("Using CPU (No GPU provider found)")
            
        return app
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace model: {e}")
        raise RuntimeError(f"Could not load InsightFace model: {e}")


class FaceDetector:
    """
    Service for detecting faces and extracting embeddings using InsightFace.
    """
    
    def __init__(self, batch_size: Optional[int] = None, enable_thumbnails: bool = True):
        """
        Initialize FaceDetector with model loading and validation.
        
        Args:
            batch_size: Number of faces to process in a batch. If None, automatically determined.
            enable_thumbnails: Whether to generate face thumbnails during extraction.
        
        Raises:
            RuntimeError: If detection or recognition models cannot be loaded.
        """
        self.app = get_face_analyser()
        self.enable_thumbnails = enable_thumbnails
        
        # Extract detection model
        self._det_model = getattr(self.app, 'det_model', None)
        
        # Find recognition model
        self._rec_model = self._find_recognition_model()
        
        # Validate models were loaded successfully
        if not self._det_model:
            raise RuntimeError("Failed to load detection model from InsightFace")
        if not self._rec_model:
            raise RuntimeError("Failed to load recognition model from InsightFace")
        
        logger.info("✅ Face detection and recognition models loaded successfully")
        
        # Set batch size
        if batch_size is None:
            self.batch_size = RECOGNITION_BATCH_SIZE
            logger.info(f"Using default batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size
            logger.info(f"Using batch size: {self.batch_size}")
        
        # Setup thumbnails directory
        self._thumbnails_dir = Path("thumbnails")
        if self.enable_thumbnails:
            self._thumbnails_dir.mkdir(exist_ok=True)
            # Clean up old thumbnails on initialization
            self.cleanup_old_thumbnails()
    
    def _find_recognition_model(self):
        """
        Find the recognition model from InsightFace app structure.
        
        Returns:
            Recognition model object or None if not found.
        """
        # Check if models is a dict (InsightFace >= 0.7)
        if isinstance(self.app.models, dict):
            for key, model in self.app.models.items():
                if any(keyword in key for keyword in ['rec', 'arcface', 'w600k', 'recognition']):
                    logger.info(f"Found recognition model: {key}")
                    return model
        
        # Check if models is a list (older versions)
        elif isinstance(self.app.models, list):
            for model in self.app.models:
                if hasattr(model, 'taskname') and model.taskname == 'recognition':
                    logger.info(f"Found recognition model with taskname: {model.taskname}")
                    return model
        
        logger.error("Could not find recognition model in InsightFace structure")
        return None
    
    def _get_optimal_batch_size(self) -> int:
        """
        Detect optimal batch size based on available GPU memory.
        
        Returns:
            Optimal batch size between MIN_BATCH_SIZE and MAX_BATCH_SIZE.
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0]  # Get free memory in bytes
                # Estimate: ~50MB per face embedding
                optimal = min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, free_mem // (FACE_MEMORY_ESTIMATE_MB * 1024 * 1024)))
                logger.info(f"Calculated optimal batch size: {optimal} (based on {free_mem / (1024**3):.2f}GB free GPU memory)")
                return optimal
        except Exception as e:
            logger.debug(f"Could not detect optimal batch size: {e}")
        
        logger.info(f"Using default batch size: {DEFAULT_BATCH_SIZE}")
        return DEFAULT_BATCH_SIZE
    
    def _validate_inputs(self, imgs: List[np.ndarray], filenames: List[str]):
        """
        Validate input images and filenames.
        
        Args:
            imgs: List of OpenCV images
            filenames: List of filenames
        
        Raises:
            ValueError: If inputs are invalid.
        """
        if len(imgs) != len(filenames):
            raise ValueError(f"Image count ({len(imgs)}) != filename count ({len(filenames)})")
        
        for i, (img, filename) in enumerate(zip(imgs, filenames)):
            if img is None or img.size == 0:
                raise ValueError(f"Image {i} ({filename}) is empty or None")
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Image {i} ({filename}) is not in BGR format (shape: {img.shape})")
    
    def extract_embeddings(
        self,
        imgs: List[np.ndarray],
        filenames: List[str]
    ) -> List[FaceEmbedding]:
        """
        Extract face embeddings from a batch of images.
        
        Args:
            imgs: List of OpenCV images (BGR format)
            filenames: List of original filenames for reference
            
        Returns:
            List of FaceEmbedding objects from all images
        
        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        self._validate_inputs(imgs, filenames)
        
        all_embeddings = []

        start_time = time.time()
        det_time = 0
        rec_time = 0
        
        # --- OPTIMIZED BATCH PIPELINE (Phase 3) ---
        
        # 1. SERIAL DETECTION + ACCUMULATION
        all_detected_faces = [] # List of {'img', 'face', 'filename', 'idx'}
        
        for img_idx, (img, filename) in enumerate(zip(imgs, filenames)):
            try:
                t0 = time.time()
                # Run Detection
                faces, kpss = self._det_model.detect(img, max_num=0, metric='default')

                t1 = time.time()
                det_time += (t1 - t0)
                
                if faces.shape[0] == 0:
                    logger.debug(f"No faces detected in {filename}")
                    continue
                     
                for i in range(faces.shape[0]):
                    bbox = faces[i, 0:4]
                    det_score = faces[i, 4]
                    kps = None if kpss is None else kpss[i]
                    
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    
                    # Fast Pre-Filter (Avoid keeping bad faces)
                    if face.det_score < DET_CONF_THRESH:
                        logger.debug(f"Low confidence face in {filename} (score: {face.det_score:.3f})")
                        continue
                    
                    box = face.bbox.astype(int)
                    if (box[2]-box[0]) <= 0 or (box[3]-box[1]) <= 0:
                        logger.warning(f"Invalid bounding box in {filename}: {box}")
                        continue
                    
                    all_detected_faces.append({
                        'img': img, 
                        'face': face,
                        'filename': filename,
                        'idx': img_idx
                    })
            except Exception as e:
                logger.warning(f"Error detecting faces in {filename}: {e}")
                continue

        # 2. BATCH RECOGNITION (The Big Speedup)
        if all_detected_faces:
            t2 = time.time()
            
            # Prepare Alignment Crops
            from insightface.utils import face_align
            
            crops = []
            valid_indices = []
            
            for i, item in enumerate(all_detected_faces):
                try:
                    kps = item['face'].kps
                    # norm_crop aligns face to standard size for ArcFace
                    aimg = face_align.norm_crop(item['img'], landmark=kps, image_size=FACE_ALIGNMENT_SIZE)
                    crops.append(aimg)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to align face in {item['filename']}: {e}")
                    pass
            
            if crops:
                # Process in mini-batches to fit in GPU VRAM
                total_crops = len(crops)
                all_feats = []
                
                for i in range(0, total_crops, self.batch_size):
                    batch_crops = crops[i:i+self.batch_size]
                    
                    # RUN INFERENCE
                    # get_feat returns (N, 512) embeddings
                    feats = self._rec_model.get_feat(batch_crops) 
                    
                    # Robust handling of return types
                    if isinstance(feats, list): 
                        feats = feats[0]
                    if len(feats.shape) == 1: # Single result case
                         feats = [feats]
                    all_feats.extend(feats)
                
                # Assign embeddings back to faces
                for i, feat in zip(valid_indices, all_feats):
                    all_detected_faces[i]['face'].embedding = feat.flatten()
            
            t3 = time.time()
            rec_time += (t3 - t2)
            
        # 3. GENERATE OUTPUTS (Thumbnails & Embedding Objects)
        # Using a per-file counter for thumbnail naming
        file_face_counters = {}
        
        for item in all_detected_faces:
            face = item['face']
            if face.embedding is None:
                logger.debug(f"Skipping face without embedding in {item['filename']}")
                continue # Skip if recognition failed
            
            filename = item['filename']
            img = item['img']
            
            # Update counter
            f_idx = file_face_counters.get(filename, 0)
            file_face_counters[filename] = f_idx + 1
            
            # Geometry
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w_box = x2 - x1
            h_box = y2 - y1
            
            # Thumbnail generation (if enabled)
            thumb_path = None
            if self.enable_thumbnails:
                h_img, w_img, _ = img.shape
                cx, cy = x1 + w_box // 2, y1 + h_box // 2
                max_dim = max(w_box, h_box)
                context_size = int(max_dim * THUMBNAIL_CONTEXT_MULTIPLIER)
                half_size = context_size // 2
                tx1, ty1 = max(0, cx - half_size), max(0, cy - half_size)
                tx2, ty2 = min(w_img, cx + half_size), min(h_img, cy + half_size)
                
                face_crop = img[ty1:ty2, tx1:tx2]
                
                if face_crop.size > 0:
                    # Convert BGR to RGB - browsers/JPEG viewers expect RGB order
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    safe_filename = filename.replace('.', '_')
                    thumb_filename = f"thumb_{safe_filename}_{f_idx}.jpg"
                    thumb_path = str(self._thumbnails_dir / thumb_filename)
                    # Save RGB data
                    cv2.imwrite(thumb_path, face_crop_rgb)

            all_embeddings.append(
                FaceEmbedding(
                    embedding=face.embedding,
                    filename=filename,
                    face_index=f_idx,
                    confidence=float(face.det_score),
                    facial_area={'x': int(x1), 'y': int(y1), 'w': int(w_box), 'h': int(h_box)},
                    landmarks=face.kps,
                    thumbnail_path=thumb_path
                )
            )
        
        # Log profiling stats for this batch
        total_batch_time = time.time() - start_time
        if total_batch_time > 0:
            logger.info(f"🛑 GPU Profiling (Batch): Detect={det_time:.3f}s ({(det_time/total_batch_time)*100:.1f}%), Rec={rec_time:.3f}s ({(rec_time/total_batch_time)*100:.1f}%)")
            
        return all_embeddings
    
    def cleanup_old_thumbnails(self, max_age_hours: int = DEFAULT_THUMBNAIL_MAX_AGE_HOURS):
        """
        Remove thumbnails older than max_age_hours.
        
        Args:
            max_age_hours: Maximum age in hours for thumbnails to keep.
        """
        if not self._thumbnails_dir.exists():
            return
        
        import time
        now = time.time()
        removed_count = 0
        
        for f in self._thumbnails_dir.iterdir():
            if f.is_file():
                try:
                    if now - f.stat().st_mtime > max_age_hours * 3600:
                        f.unlink()
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove old thumbnail {f}: {e}")
        
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} old thumbnails")
    
    def cleanup(self):
        """Cleanup resources if needed."""
        # Models are cached, nothing specific to clean
        pass
