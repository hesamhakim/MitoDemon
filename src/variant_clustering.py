"""
Variant Clustering Module for Single-Cell Mitochondrial Analysis
Author: [Your Name]
Date: 2025
Description: Clusters mitochondrial variants into representative groups
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional, Dict, Any
import warnings

class VariantClusterer:
    """Cluster mitochondrial variants into representative groups"""
    
    def __init__(self, method: str = 'spectral', n_clusters: int = 10, random_state: int = 42, use_pca: bool = True, pca_components: int = 50):
        """
        Initialize clusterer
        
        Parameters:
        -----------
        method : str
            Clustering method ('kmeans', 'dbscan', 'hierarchical', 'nmf', 'spectral' - new default)
        n_clusters : int
            Number of clusters (for methods that require it)
        random_state : int
            Random seed for reproducibility
        use_pca : bool
            Apply PCA for dimensionality reduction before clustering (new: for sparsity handling)
        pca_components : int
            Number of PCA components (new)
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.cluster_labels = None
        self.cluster_centers = None
        
    def fit_predict(self, variant_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit clustering model and predict cluster assignments
        
        Parameters:
        -----------
        variant_matrix : np.ndarray
            Variant matrix (positions x UMIs, continuous or binary - now handles continuous)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            cluster_signatures: Continuous matrix (positions x clusters)
            cluster_assignments: Cluster label for each UMI
        """
        # Transpose for clustering (samples as rows)
        X = variant_matrix.T
        
        # Remove all-zero samples to prevent NaN issues in clustering
        non_zero_mask = np.any(X != 0, axis=1)
        n_non_zero = np.sum(non_zero_mask)
        if n_non_zero == 0:
            warnings.warn("All samples are zero vectors. Assigning all to cluster 0.")
            return np.zeros((variant_matrix.shape[0], 1)), np.zeros(variant_matrix.shape[1], dtype=int)
        if n_non_zero < self.n_clusters:
            warnings.warn(f"Too few non-zero samples ({n_non_zero}) for {self.n_clusters} clusters. Falling back to kmeans with reduced clusters.")
            self.n_clusters = max(1, n_non_zero // 2)
            self.method = 'kmeans'
        X = X[non_zero_mask]
        
        # Apply PCA if enabled (new: for dimensionality reduction on sparse data)
        if self.use_pca and X.shape[0] > 1 and X.shape[1] > 1:
            n_comp = min(self.pca_components, min(X.shape[0]-1, X.shape[1]))
            pca = PCA(n_components=n_comp)
            X = pca.fit_transform(X)
            print(f"Applied PCA: Reduced to {X.shape[1]} components")
        
        if self.method == 'kmeans':
            cluster_labels, cluster_centers = self._kmeans_clustering(X)
        elif self.method == 'nmf':
            cluster_labels, cluster_centers = self._nmf_clustering(variant_matrix)  # NMF on original
        elif self.method == 'hierarchical':
            cluster_labels, cluster_centers = self._hierarchical_clustering(X)
        elif self.method == 'dbscan':
            cluster_labels, cluster_centers = self._dbscan_clustering(X)
        elif self.method == 'spectral':
            cluster_labels, cluster_centers = self._spectral_clustering(X)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Map labels back to original size (assign -1 to zero vectors)
        full_labels = np.full(variant_matrix.shape[1], -1)
        full_labels[non_zero_mask] = cluster_labels
        self.cluster_labels = full_labels
        
        self.cluster_centers = cluster_centers
        
        # Create continuous cluster signatures (mean VAF per cluster)
        cluster_signatures = self._create_cluster_signatures(variant_matrix, full_labels, use_binary=False)
        
        return cluster_signatures, full_labels
    
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
        from sklearn.metrics.pairwise import cosine_distances
        
        distances = cosine_distances(X)
        
        # Estimate eps
        k = min(5, X.shape[0] - 1)
        k_distances = np.sort(distances, axis=1)[:, k]
        eps = np.percentile(k_distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distances)
        
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
        
        # Reassign noise points
        if -1 in unique_labels:
            noise_mask = cluster_labels == -1
            for idx in np.where(noise_mask)[0]:
                distances_to_centers = np.sum((cluster_centers - X[idx])**2, axis=1)
                cluster_labels[idx] = np.argmin(distances_to_centers)
        
        return cluster_labels, cluster_centers
    
    def _spectral_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Spectral clustering with RBF affinity (revised: to handle potential negative similarities and NaNs)"""
        # Handle potential NaNs after PCA or in data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='rbf',  # Revised: use RBF instead of cosine to ensure positive affinities and avoid NaN issues
            random_state=self.random_state
        )
        cluster_labels = clustering.fit_predict(X)
    
        # Calculate cluster centers
        cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            if mask.sum() > 0:
                cluster_centers[i] = X[mask].mean(axis=0)
    
        return cluster_labels, cluster_centers
    
    def _create_cluster_signatures(self, variant_matrix: np.ndarray, 
                                  cluster_labels: np.ndarray,
                                  use_binary: bool = False) -> np.ndarray:
        """
        Create cluster signatures with continuous values (mean VAF)
        """
        n_positions = variant_matrix.shape[0]
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))  # Ignore -1 if present
    
        cluster_signatures = np.zeros((n_positions, n_clusters))
    
        # Remap labels to 0 to n_clusters-1 (ignoring -1)
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        label_map = {old: new for new, old in enumerate(unique_labels)}
    
        for old_id in unique_labels:
            cluster_id = label_map[old_id]
            cluster_mask = cluster_labels == old_id
            if cluster_mask.sum() > 0:
                # Use mean VAF (continuous, no threshold)
                cluster_signatures[:, cluster_id] = variant_matrix[:, cluster_mask].mean(axis=1)
            
                if use_binary:
                    cluster_signatures[:, cluster_id] = (cluster_signatures[:, cluster_id] >= 0.5).astype(int)
    
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
        
        # Ignore -1 labels for stats
        valid_mask = self.cluster_labels >= 0
        valid_labels = self.cluster_labels[valid_mask]
        
        stats = {
            'n_clusters': len(np.unique(valid_labels)),
            'cluster_sizes': np.bincount(valid_labels),
            'variants_per_cluster': []
        }
        
        # Calculate mean VAF per cluster (new: for continuous)
        for cluster_id in np.unique(valid_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_vaf = variant_matrix[:, cluster_mask].mean() if cluster_mask.sum() > 0 else 0
            stats['variants_per_cluster'].append(cluster_vaf)
        
        stats['variants_per_cluster'] = np.array(stats['variants_per_cluster'])
        stats['mean_variants_per_cluster'] = stats['variants_per_cluster'].mean()
        stats['std_variants_per_cluster'] = stats['variants_per_cluster'].std()
        
        return stats
