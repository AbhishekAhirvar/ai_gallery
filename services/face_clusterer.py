"""
Face Clustering Service for VowScan.

Groups face embeddings into person clusters using DBSCAN with cosine distance.
Automatically selects the best thumbnail for each cluster based on face quality metrics.

Example usage:
    from services.face_clusterer import FaceClusterer
    from services.face_detector import FaceDetector
    
    detector = FaceDetector()
    clusterer = FaceClusterer()
    
    # Detect faces in images
    embeddings = []
    for image_path in image_paths:
        faces = detector.detect_faces(image_path)
        embeddings.extend(faces)
    
    # Cluster faces into person groups
    clusters, unclustered = clusterer.cluster(embeddings)
    
    # Access results
    for cluster in clusters:
        print(f"{cluster.label}: {cluster.photo_count} photos")
        print(f"Face counts per file: {cluster.face_count_per_file}")
        print(f"Best thumbnail: {cluster.thumbnail_path}")
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from config import (
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    THUMBNAIL_SCAN_LIMIT,
    FACE_SIZE_QUALITY_THRESH,
    YAW_FRONTALITY_THRESH,
    CONFIDENCE_QUALITY_THRESH
)
from services.face_detector import FaceEmbedding


@dataclass
class PersonCluster:
    """
    Represents a cluster of faces belonging to one person.
    
    Attributes:
        label: Human-readable label (e.g., "Person 1")
        filenames: List of unique filenames containing this person
        thumbnail_path: Path to the best representative face image
        embeddings: List of all FaceEmbedding objects in this cluster
    """
    label: str
    filenames: List[str]
    thumbnail_path: Optional[str] = None
    embeddings: Optional[List[FaceEmbedding]] = field(default=None)
    
    @property
    def photo_count(self) -> int:
        """Number of unique photos containing this person."""
        return len(self.filenames)
    
    @property
    def face_count_per_file(self) -> Dict[str, int]:
        """
        Returns dictionary mapping filename to count of faces detected.
        
        Useful for identifying photos with multiple instances of the same person
        (e.g., mirror selfies, group photos with duplicates).
        """
        if not self.embeddings:
            return {}
        return dict(Counter(fe.filename for fe in self.embeddings))


class FaceClusterer:
    """
    Service for clustering face embeddings into person groups.
    
    Uses DBSCAN with cosine distance for automatic grouping without requiring
    the number of people to be specified in advance. The algorithm automatically
    identifies core samples (faces that appear frequently) and expands clusters
    around them while marking outliers as noise.
    
    Attributes:
        eps: Maximum distance between two samples to be considered neighbors.
             Lower values create stricter clusters. Default: 0.7
        min_samples: Minimum number of samples in a neighborhood to form a core point.
                    This effectively sets the minimum cluster size. Default: 3
        thumbnail_scan_limit: Maximum faces to evaluate when selecting best thumbnail.
                            Prevents excessive computation for large clusters. Default: 50
    """
    
    def __init__(
        self, 
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        min_samples: int = MIN_SAMPLES,
        thumbnail_scan_limit: int = THUMBNAIL_SCAN_LIMIT
    ):
        """
        Initialize the face clusterer.
        
        Args:
            min_cluster_size: HDBSCAN min_cluster_size
            min_samples: HDBSCAN min_samples
            thumbnail_scan_limit: Max faces to scan when selecting thumbnails
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.thumbnail_scan_limit = thumbnail_scan_limit

    def _validate_embeddings(self, face_embeddings: List[FaceEmbedding]) -> None:
        """
        Validate input embeddings for clustering.
        
        Args:
            face_embeddings: List of FaceEmbedding objects to validate
            
        Raises:
            ValueError: If embeddings are invalid (inconsistent dimensions, etc.)
        """
        if not face_embeddings:
            return  # Empty list is valid, will be handled by caller
        
        # Check that all embeddings have the same dimension
        expected_dim = len(face_embeddings[0].embedding)
        for i, fe in enumerate(face_embeddings):
            if fe.embedding is None:
                raise ValueError(f"FaceEmbedding at index {i} has None embedding")
            
            actual_dim = len(fe.embedding)
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions: expected {expected_dim}, "
                    f"got {actual_dim} at index {i}"
                )
        
        # Validate that embeddings are not all zeros (would indicate extraction failure)
        for i, fe in enumerate(face_embeddings):
            if np.allclose(fe.embedding, 0):
                raise ValueError(f"FaceEmbedding at index {i} is all zeros")

    def _smooth_profile_penalty(self, yaw_score: float) -> float:
        """
        Apply smooth penalty for profile faces using sigmoid function.
        
        Instead of harsh 0/1 cutoffs, uses a sigmoid to smoothly transition
        from full penalty (profile) to no penalty (frontal).
        
        The sigmoid is centered at 0.5 and has a steep slope (factor=10),
        meaning:
        - yaw_score < 0.3 → heavy penalty (~0.05)
        - yaw_score = 0.5 → moderate penalty (0.5)
        - yaw_score > 0.7 → minimal penalty (~0.95)
        
        Args:
            yaw_score: Frontality score from 0 (pure profile) to 1 (perfectly frontal)
            
        Returns:
            Penalty factor from 0.05 to 1.0
        """
        # Sigmoid: 1 / (1 + exp(-k * (x - x0)))
        # k=10 for steep transition, x0=0.5 for center
        return 1.0 / (1.0 + np.exp(-10.0 * (yaw_score - 0.5)))

    def _select_best_face(self, faces: List[FaceEmbedding]) -> Optional[str]:
        """
        Select the best face for the cluster thumbnail based on quality metrics.
        
        Scoring formula combines three factors:
        1. Size score: width × height (larger faces are usually sharper)
        2. Frontality score: symmetry of eye distances to nose (frontal > profile)
        3. Detection confidence: model's confidence in face detection
        
        Final score = size × frontality_penalty × confidence
        
        Where frontality_penalty is a smooth sigmoid function that:
        - Heavily penalizes profile faces (yaw < 0.3)
        - Moderately penalizes semi-profile (0.3 < yaw < 0.7)
        - Minimally affects frontal faces (yaw > 0.7)
        
        Early stopping: If we find an "excellent" face (large, frontal, confident),
        we stop scanning to save computation.
        
        Thresholds chosen based on empirical testing:
        - FACE_SIZE_QUALITY_THRESHOLD (15000 px²) ≈ 122×122 minimum for sharp face
        - YAW_FRONTALITY_THRESHOLD (0.8) ensures nearly head-on view
        - CONFIDENCE_QUALITY_THRESHOLD (0.7) filters out uncertain detections
        
        Args:
            faces: List of FaceEmbedding objects from the same cluster
            
        Returns:
            Path to the best thumbnail image, or None if no valid thumbnails found
        """
        if not faces:
            return None
        
        # Fallback: use first available thumbnail
        best_path = None
        for face in faces:
            if face.thumbnail_path is not None:
                best_path = face.thumbnail_path
                break
        
        if best_path is None:
            return None  # No thumbnails available at all
        
        best_score = -1
        scanned_count = 0
        
        for face in faces:
            if face.thumbnail_path is None:
                continue
            
            scanned_count += 1
            if scanned_count > self.thumbnail_scan_limit:
                break
                
            # 1. Size Score (larger faces tend to be sharper, less blurry)
            w = face.facial_area['w']
            h = face.facial_area['h']
            size_score = w * h
            
            # 2. Frontality Score (frontal faces better than profiles)
            # Using eye-to-nose distance ratio as proxy for yaw angle
            # kps indices: 0=LeftEye, 1=RightEye, 2=Nose, 3=LeftMouth, 4=RightMouth
            yaw_score = 0.5  # Default neutral score if no landmarks
            
            if face.landmarks is not None:
                kps = face.landmarks
                l_eye = kps[0]  # Left eye position
                r_eye = kps[1]  # Right eye position
                nose = kps[2]   # Nose tip position
                
                # Calculate distances from each eye to nose
                d_l = np.linalg.norm(l_eye - nose)
                d_r = np.linalg.norm(r_eye - nose)
                
                # Ratio of closer eye to farther eye
                # For frontal face: ratio ≈ 1.0 (both eyes equidistant)
                # For profile: ratio ≈ 0.0 (one eye much closer)
                yaw_score = min(d_l, d_r) / (max(d_l, d_r) + 1e-6)
                
                # TODO: Future enhancement - incorporate pitch and roll angles
                # pitch = vertical tilt (looking up/down)
                # roll = head rotation (tilted sideways)
                # May require additional landmark calculations or model outputs
            
            # 3. Apply smooth profile penalty
            yaw_factor = self._smooth_profile_penalty(yaw_score)
            
            # 4. Composite Score
            score = size_score * yaw_factor * face.confidence
            
            if score > best_score:
                best_score = score
                best_path = face.thumbnail_path
                
            # Optimization: Early stop if we found an "excellent" face
            # This saves computation for large clusters
            if (size_score > FACE_SIZE_QUALITY_THRESH and 
                yaw_score > YAW_FRONTALITY_THRESH and 
                face.confidence > CONFIDENCE_QUALITY_THRESH):
                break
                
        return best_path

    def cluster(
        self, 
        face_embeddings: List[FaceEmbedding]
    ) -> Tuple[List[PersonCluster], List[FaceEmbedding]]:
        """
        Cluster face embeddings into person groups using DBSCAN.
        
        DBSCAN automatically determines the number of clusters and handles noise.
        It groups faces based on cosine distance in the embedding space, meaning
        faces with similar features (likely the same person) are clustered together.
        
        Memory optimization: Uses sklearn's built-in cosine metric instead of
        precomputing the full distance matrix, reducing memory from O(n²) to O(n).
        
        Args:
            face_embeddings: List of FaceEmbedding objects to cluster
            
        Returns:
            Tuple of (clusters, unclustered):
                - clusters: List of PersonCluster objects, sorted by size (largest first)
                - unclustered: List of FaceEmbedding objects that didn't form clusters
                              (noise points according to DBSCAN)
                              
        Raises:
            ValueError: If embeddings have inconsistent dimensions or are invalid
            
        Example:
            >>> clusterer = FaceClusterer(eps=0.7, min_samples=3)
            >>> clusters, noise = clusterer.cluster(all_faces)
            >>> print(f"Found {len(clusters)} people in photos")
            >>> for cluster in clusters:
            ...     print(f"{cluster.label}: appears in {cluster.photo_count} photos")
        """
        # Handle empty input
        if not face_embeddings:
            return [], []
        
        # Validate inputs
        self._validate_embeddings(face_embeddings)
        
        # Extract embeddings array
        embeddings = np.array([fe.embedding for fe in face_embeddings])
        
        # Choose clustering approach based on config
        from config import USE_FAISS_CLUSTERING, FAISS_K_NEIGHBORS
        
        if USE_FAISS_CLUSTERING:
            # Use hybrid FAISS + HDBSCAN for faster clustering
            from services.faiss_clusterer import HybridFAISSClusterer
            
            clusterer = HybridFAISSClusterer(
                k_neighbors=FAISS_K_NEIGHBORS,
                min_cluster_size=self.min_cluster_size,
                min_samples=1,
                metric='cosine',
                use_gpu=False  # CPU is sufficient for our dataset size
            )
            labels, probabilities = clusterer.cluster(embeddings)
        else:
            # Use pure HDBSCAN (Robust and better than DBSCAN)
            import hdbscan
            
            # Use 'euclidean' for normalized vectors (equivalent to cosine)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=1,
                metric='euclidean', 
                cluster_selection_method='eom',
                allow_single_cluster=False
            )
            labels = clusterer.fit_predict(embeddings)
        
        # Group faces by cluster label
        cluster_data: Dict[int, List[FaceEmbedding]] = {}
        unclustered_embeddings: List[FaceEmbedding] = []
        
        for i, label in enumerate(labels):
            fe = face_embeddings[i]
            if label == -1:
                # DBSCAN marks outliers/noise with label -1
                unclustered_embeddings.append(fe)
            else:
                if label not in cluster_data:
                    cluster_data[label] = []
                cluster_data[label].append(fe)
        
        # Build result list
        clusters = []
        person_idx = 1
        
        # Sort clusters by size (largest first) for better UX
        sorted_labels = sorted(
            cluster_data.keys(),
            key=lambda l: len(cluster_data[l]),
            reverse=True
        )
        
        for label in sorted_labels:
            group_embeddings = cluster_data[label]
            
            # Get unique filenames (one person may appear multiple times per photo)
            unique_filenames = list(set(fe.filename for fe in group_embeddings))
            
            # Note: We removed the redundant min_samples check here because DBSCAN
            # already enforces this constraint internally. Any cluster that reaches
            # this point is guaranteed to have >= min_samples members.
            
            # Select best thumbnail (frontal, sharp, large, confident)
            thumb_path = self._select_best_face(group_embeddings)
            
            clusters.append(PersonCluster(
                label=f"Person {person_idx}",
                filenames=unique_filenames,
                thumbnail_path=thumb_path,
                embeddings=group_embeddings  # Store for face_count_per_file property
            ))
            person_idx += 1
        
        return clusters, unclustered_embeddings
