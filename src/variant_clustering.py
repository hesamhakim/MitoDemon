"""
Variant Clustering Module for Single-Cell Mitochondrial Analysis
Author: [Your Name]
Date: 2025
Description: Clusters mitochondrial variants into representative groups
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any
import warnings

class VariantClusterer:
    """Cluster mitochondrial variants into representative groups"""
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 10, random_state: int = 42):
        """
        Initialize clusterer
        
        Parameters:
        -----------
        method : str
            Clustering method ('kmeans', 'dbscan', 'hierarchical', 'nmf')
        n_clusters : int
            Number of clusters (for methods that require it)
        random_state : int
            Random seed for reproducibility
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit_predict(self, variant_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit clustering model and predict cluster assignments
        
        Parameters:
        -----------
        variant_matrix : np.ndarray
            Binary variant matrix (positions x UMIs)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            cluster_signatures: Binary matrix (positions x clusters)
            cluster_assignments: Cluster label for each UMI
        """
        # Transpose for clustering (samples as rows)
        X = variant_matrix.T
        
        if self.method == 'kmeans':
            cluster_labels, cluster_centers = self._kmeans_clustering(X)
        elif self.method == 'nmf':
            cluster_labels, cluster_centers = self._nmf_clustering(variant_matrix)
        elif self.method == 'hierarchical':
            cluster_labels, cluster_centers = self._hierarchical_clustering(X)
        elif self.method == 'dbscan':
            cluster_labels, cluster_centers = self._dbscan_clustering(X)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers
        
        # Create cluster signatures (binary matrix)
        cluster_signatures = self._create_cluster_signatures(variant_matrix, cluster_labels)
        
        return cluster_signatures, cluster_labels
    
    def _kmeans_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """K-means clustering"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_
        return cluster_labels, cluster_centers
    
    def _nmf_clustering(self, variant_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Non-negative Matrix Factorization clustering"""
        nmf = NMF(n_components=self.n_clusters, random_state=self.random_state, max_iter=500)
        W = nmf.fit_transform(variant_matrix)  # positions x components
        H = nmf.components_  # components x UMIs
        
        # Assign each UMI to cluster with highest weight
        cluster_labels = np.argmax(H, axis=0)
        cluster_centers = W.T  # transpose to match other methods
        
        return cluster_labels, cluster_centers
    
    def _hierarchical_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Agglomerative hierarchical clustering"""
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters)
        cluster_labels = clustering.fit_predict(X)
        
        # Calculate cluster centers
        cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            if mask.sum() > 0:
                cluster_centers[i] = X[mask].mean(axis=0)
        
        return cluster_labels, cluster_centers
    
    def _dbscan_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """DBSCAN clustering"""
        # Use cosine similarity for binary data
        from sklearn.metrics.pairwise import cosine_distances
        
        # Calculate pairwise distances
        distances = cosine_distances(X)
        
        # Estimate eps using k-distance graph
        k = min(5, X.shape[0] - 1)
        k_distances = np.sort(distances, axis=1)[:, k]
        eps = np.percentile(k_distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distances)
        
        # Handle noise points (-1 labels)
        unique_labels = np.unique(cluster_labels)
        n_clusters_found = len(unique_labels[unique_labels >= 0])
        
        if n_clusters_found == 0:
            warnings.warn("DBSCAN found no clusters. Falling back to k-means.")
            return self._kmeans_clustering(X)
        
        # Calculate cluster centers
        cluster_centers = []
        for label in unique_labels[unique_labels >= 0]:
            mask = cluster_labels == label
            cluster_centers.append(X[mask].mean(axis=0))
        
        cluster_centers = np.array(cluster_centers)
        
        # Reassign noise points to nearest cluster
        if -1 in unique_labels:
            noise_mask = cluster_labels == -1
            for idx in np.where(noise_mask)[0]:
                distances_to_centers = np.sum((cluster_centers - X[idx])**2, axis=1)
                cluster_labels[idx] = np.argmin(distances_to_centers)
        
        return cluster_labels, cluster_centers
    
    def _create_cluster_signatures(self, variant_matrix: np.ndarray, 
                                  cluster_labels: np.ndarray) -> np.ndarray:
        """
        Create binary cluster signatures
        
        Parameters:
        -----------
        variant_matrix : np.ndarray
            Original variant matrix (positions x UMIs)
        cluster_labels : np.ndarray
            Cluster assignment for each UMI
        
        Returns:
        --------
        np.ndarray
            Binary cluster signatures (positions x clusters)
        """
        n_positions = variant_matrix.shape[0]
        n_clusters = len(np.unique(cluster_labels))
        
        cluster_signatures = np.zeros((n_positions, n_clusters))
        
        for cluster_id in range(n_clusters):
            # Get UMIs in this cluster
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() > 0:
                # Average variants across UMIs in cluster
                cluster_variants = variant_matrix[:, cluster_mask].mean(axis=1)
                # Binarize (threshold at 0.5)
                cluster_signatures[:, cluster_id] = (cluster_variants >= 0.5).astype(int)
        
        return cluster_signatures
    
    def get_cluster_statistics(self, variant_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the clustering results
        
        Parameters:
        -----------
        variant_matrix : np.ndarray
            Original variant matrix
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing clustering statistics
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed yet. Run fit_predict() first.")
        
        stats = {
            'n_clusters': len(np.unique(self.cluster_labels)),
            'cluster_sizes': np.bincount(self.cluster_labels),
            'variants_per_cluster': []
        }
        
        # Calculate variants per cluster
        for cluster_id in np.unique(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_variants = variant_matrix[:, cluster_mask].sum()
            stats['variants_per_cluster'].append(cluster_variants)
        
        stats['variants_per_cluster'] = np.array(stats['variants_per_cluster'])
        stats['mean_variants_per_cluster'] = stats['variants_per_cluster'].mean()
        stats['std_variants_per_cluster'] = stats['variants_per_cluster'].std()
        
        return stats
