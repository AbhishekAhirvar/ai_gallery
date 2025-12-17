"""
Face Clustering Service for VowScan.
Groups face embeddings into person clusters using DBSCAN.
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES
from services.face_detector import FaceEmbedding


@dataclass
class PersonCluster:
    """Represents a cluster of faces belonging to one person."""
    label: str
    filenames: List[str]
    thumbnail_path: str = None
    embeddings: List[FaceEmbedding] = None
    
    @property
    def photo_count(self) -> int:
        return len(self.filenames)


class FaceClusterer:
    """
    Service for clustering face embeddings into person groups.
    Uses DBSCAN with cosine distance for automatic grouping.
    """
    
    def __init__(self, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES):
        self.eps = eps
        self.min_samples = min_samples

    def _select_best_face(self, faces: List[FaceEmbedding]) -> str:
        """
        Selects the best face for the thumbnail based on frontality, size, and confidence.
        """
        best_score = -1
        best_path = None
        
        # Fallback
        if not faces: return None
        best_path = faces[0].thumbnail_path
        
        for face in faces:
            if face.thumbnail_path is None: continue
            
            # 1. Size Score (Larger is better, usually less blurry)
            w = face.facial_area['w']
            h = face.facial_area['h']
            size_score = (w * h)
            
            # 2. Frontality Score (Symmetry of eyes relative to nose)
            # kps: 0=LeftEye, 1=RightEye, 2=Nose
            yaw_score = 0.5 # Default if no landmarks
            if face.landmarks is not None:
                kps = face.landmarks
                l_eye = kps[0]
                r_eye = kps[1]
                nose = kps[2]
                
                d_l = np.linalg.norm(l_eye - nose)
                d_r = np.linalg.norm(r_eye - nose)
                
                # Ratio of closer eye dist to further eye dist
                # If perfectly frontal, d_l == d_r -> ratio = 1.0
                # If side profile, one is very small -> ratio -> 0.0
                yaw_score = min(d_l, d_r) / (max(d_l, d_r) + 1e-6)
            
            # 3. Composite Score
            # We prioritize Frontality first, then Size.
            # Yaw is 0.0-1.0. Size is pixels (e.g. 10000 - 40000).
            # We want to penalize bad yaw heavily.
            
            # If yaw < 0.4, it's a hard profile. Penalize.
            yaw_factor = yaw_score if yaw_score > 0.4 else yaw_score * 0.1
            
            score = size_score * yaw_factor * face.confidence
            
            if score > best_score:
                best_score = score
                best_path = face.thumbnail_path
                
        return best_path
    
    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine distance matrix from embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)
        
        # Convert to distance (1 - similarity)
        distance = 1 - similarity
        
        # Ensure valid range [0, 2]
        return np.clip(distance, 0, 2)
    
    def cluster(self, face_embeddings: List[FaceEmbedding]) -> Tuple[List[PersonCluster], List[FaceEmbedding]]:
        """
        Cluster face embeddings into person groups.
        
        Args:
            face_embeddings: List of FaceEmbedding objects
            
        Returns:
            Tuple: (List of PersonCluster, List of unclustered FaceEmbedding objects)
        """
        if not face_embeddings:
            return [], []
        
        # Extract embeddings and filenames
        embeddings = np.array([fe.embedding for fe in face_embeddings])
        filenames = [fe.filename for fe in face_embeddings]
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(embeddings)
        
        # Run DBSCAN
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Group by cluster label
        cluster_data: Dict[int, List[FaceEmbedding]] = {}
        unclustered_embeddings: List[FaceEmbedding] = []
        
        for i, label in enumerate(labels):
            fe = face_embeddings[i]
            if label == -1:
                unclustered_embeddings.append(fe)
            else:
                if label not in cluster_data:
                    cluster_data[label] = []
                cluster_data[label].append(fe)
        
        # Build result list
        clusters = []
        person_idx = 1
        
        # Sort clusters by size
        sorted_labels = sorted(
            cluster_data.keys(),
            key=lambda l: len(cluster_data[l]),
            reverse=True
        )
        
        for label in sorted_labels:
            group_embeddings = cluster_data[label]
            unique_filenames = list(set(fe.filename for fe in group_embeddings))
            
            # Additional check: Min samples check
            if len(unique_filenames) < self.min_samples:
                # Treat as unclustered
                unclustered_embeddings.extend(group_embeddings)
                continue
                
            # Pick best thumbnail (Frontal, Sharp, Large)
            thumb_path = self._select_best_face(group_embeddings)
            
            clusters.append(PersonCluster(
                label=f"Person {person_idx}",
                filenames=unique_filenames,
                thumbnail_path=thumb_path,
                embeddings=group_embeddings
            ))
            person_idx += 1
        
        return clusters, unclustered_embeddings

