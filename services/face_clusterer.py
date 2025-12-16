"""
Face Clustering Service for VowScan.
Groups face embeddings into person clusters using DBSCAN.
"""

from typing import List, Dict, Set
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
    
    def cluster(self, face_embeddings: List[FaceEmbedding]) -> List[PersonCluster]:
        """
        Cluster face embeddings into person groups.
        
        Args:
            face_embeddings: List of FaceEmbedding objects
            
        Returns:
            List of PersonCluster objects, sorted by cluster size
        """
        if not face_embeddings:
            return []
        
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
        cluster_files: Dict[int, Set[str]] = {}
        noise_files: List[str] = []
        
        for filename, label in zip(filenames, labels):
            if label == -1:
                # Noise point - treat as individual
                noise_files.append(filename)
            else:
                if label not in cluster_files:
                    cluster_files[label] = set()
                cluster_files[label].add(filename)
        
        # Build result list
        clusters = []
        person_idx = 1
        
        # Add proper clusters first (sorted by size, descending)
        sorted_clusters = sorted(
            cluster_files.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for _, files in sorted_clusters:
            clusters.append(PersonCluster(
                label=f"Person {person_idx}",
                filenames=list(files)
            ))
            person_idx += 1
        
        # Add noise points as individual persons
        for filename in noise_files:
            # Check if this file is already in a cluster (from multiple faces)
            already_clustered = any(
                filename in c.filenames for c in clusters
            )
            if not already_clustered:
                clusters.append(PersonCluster(
                    label=f"Person {person_idx}",
                    filenames=[filename]
                ))
                person_idx += 1
        
        return clusters
