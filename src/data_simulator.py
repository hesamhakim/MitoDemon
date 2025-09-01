"""
Data Simulation Module for Single-Cell Mitochondrial Analysis
Following the same workflow as real data processing: variants → clustering → signatures → cells

This module simulates mitochondrial variant data using a realistic workflow:
1. Generate random individual variants (simulating UMIs/molecules)
2. Create variant matrix (positions × variants)
3. Apply clustering to group similar variants
4. Extract cluster signatures from clustered variants
5. Generate cell populations based on discovered clusters
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import warnings
from .variant_clustering import VariantClusterer


class RealisticDataSimulator:
    """
    Simulate mitochondrial variant data following real data processing workflow
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_realistic_data(self, 
                               n_locations: int = 1650,
                               n_variants: int = 1000,
                               n_clusters: int = 50,
                               n_cells: int = 100,
                               min_variants_per_molecule: int = 1,
                               max_variants_per_molecule: int = 20,
                               clustering_method: str = 'kmeans',
                               homogeneity_alpha: float = 100.0,
                               noise_level: float = 0.01,
                               variant_frequency_shape: float = 2.0) -> Dict[str, Any]:
        """
        Generate realistic mitochondrial variant data following the iMiGseq workflow
        
        Parameters:
        -----------
        n_locations : int
            Number of mitochondrial positions (like 16569 for full mtDNA)
        n_variants : int
            Number of individual variant molecules to simulate (like UMIs)
        n_clusters : int
            Target number of clusters for K-means
        n_cells : int
            Number of cells to simulate
        min_variants_per_molecule : int
            Minimum variants per molecule
        max_variants_per_molecule : int
            Maximum variants per molecule
        clustering_method : str
            Clustering method ('kmeans', 'hierarchical', 'dbscan', 'nmf')
        homogeneity_alpha : float
            Controls cell-to-cell variation (higher = more homogeneous)
        noise_level : float
            Sequencing noise level
        variant_frequency_shape : float
            Controls frequency distribution of variants across genome
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all simulation results
        """
        
        print("=== REALISTIC DATA SIMULATION ===")
        print("Following iMiGseq workflow: variants → clustering → signatures → cells")
        print()
        
        # Step 1: Generate individual variant molecules (like UMIs)
        print("Step 1: Generating individual variant molecules...")
        variant_matrix = self._generate_variant_molecules(
            n_locations, n_variants, min_variants_per_molecule, 
            max_variants_per_molecule, variant_frequency_shape
        )
        
        print(f"  - Generated {n_variants} variant molecules")
        print(f"  - Variant matrix shape: {variant_matrix.shape}")
        print(f"  - Total variants: {variant_matrix.sum()}")
        print(f"  - Sparsity: {1 - variant_matrix.sum() / variant_matrix.size:.4f}")
        
        # Step 2: Apply clustering to discover patterns
        print(f"\nStep 2: Clustering variants using {clustering_method}...")
        clusterer = VariantClusterer(
            method=clustering_method, 
            n_clusters=n_clusters, 
            random_state=self.random_state
        )
        
        cluster_signatures, cluster_labels = clusterer.fit_predict(variant_matrix)
        cluster_stats = clusterer.get_cluster_statistics(variant_matrix)
        
        print(f"  - Discovered {cluster_stats['n_clusters']} clusters")
        print(f"  - Cluster sizes: {cluster_stats['cluster_sizes']}")
        print(f"  - Mean variants per cluster: {cluster_stats['mean_variants_per_cluster']:.2f}")
        
        # Step 3: Calculate initial proportions from clustering results
        print("\nStep 3: Calculating cluster proportions from discovered patterns...")
        initial_proportions = np.bincount(cluster_labels, minlength=n_clusters) / len(cluster_labels)
        initial_proportions = initial_proportions.reshape(1, -1)
        
        print(f"  - Initial proportions shape: {initial_proportions.shape}")
        print(f"  - Non-zero clusters: {(initial_proportions[0] > 0).sum()}")
        
        # Step 4: Generate cell population based on discovered clusters
        print(f"\nStep 4: Simulating {n_cells} cells based on discovered patterns...")
        cell_proportions = self._simulate_cell_population(
            initial_proportions, n_cells, homogeneity_alpha
        )
        
        # Step 5: Generate observed data with noise
        print("\nStep 5: Adding sequencing noise to create observed data...")
        C_true = cell_proportions @ cluster_signatures.T
        C_observed = C_true + np.random.normal(0, noise_level, size=C_true.shape)
        C_observed = np.clip(C_observed, 0, 1)
        
        print(f"  - Observed data shape: {C_observed.shape}")
        print(f"  - Value range: [{C_observed.min():.4f}, {C_observed.max():.4f}]")
        
        # Compile results
        results = {
            'K_true': cluster_signatures,
            'P_cells_true': cell_proportions,
            'C_observed': C_observed,
            'true_mean_proportions': initial_proportions[0],
            'num_variants_per_cluster': cluster_stats['variants_per_cluster'],
            'variant_matrix': variant_matrix,
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_stats,
            'simulation_metadata': {
                'n_locations': n_locations,
                'n_variants': n_variants,
                'n_clusters': n_clusters,
                'n_cells': n_cells,
                'clustering_method': clustering_method,
                'homogeneity_alpha': homogeneity_alpha,
                'noise_level': noise_level,
                'random_state': self.random_state
            }
        }
        
        print("\n✓ Realistic data simulation complete!")
        return results
    
    def _generate_variant_molecules(self, 
                                   n_locations: int, 
                                   n_variants: int,
                                   min_variants: int,
                                   max_variants: int,
                                   frequency_shape: float) -> np.ndarray:
        """
        Generate individual variant molecules (simulating UMIs)
        
        Each molecule has a random number of variants at random positions,
        with some positions being more likely to have variants (hotspots)
        """
        
        # Create position weights (some positions more likely to have variants)
        position_weights = np.random.gamma(frequency_shape, 1, n_locations)
        position_weights = position_weights / position_weights.sum()
        
        variant_matrix = np.zeros((n_locations, n_variants))
        
        for i in range(n_variants):
            # Random number of variants for this molecule
            n_vars_this_molecule = np.random.randint(min_variants, max_variants + 1)
            
            # Select positions based on weights (hotspots more likely)
            variant_positions = np.random.choice(
                n_locations, 
                size=n_vars_this_molecule, 
                replace=False, 
                p=position_weights
            )
            
            variant_matrix[variant_positions, i] = 1
            
        return variant_matrix
    
    def _simulate_cell_population(self, 
                                 initial_proportions: np.ndarray,
                                 n_cells: int,
                                 homogeneity_alpha: float) -> np.ndarray:
        """
        Simulate cell population based on discovered cluster proportions
        
        Uses Dirichlet distribution to create variations around the mean proportions
        """
        
        cell_proportions_list = []
        
        # Ensure no zero proportions for Dirichlet
        safe_proportions = np.where(initial_proportions[0] == 0, 1e-6, initial_proportions[0])
        
        for _ in range(n_cells):
            # Generate cell-specific proportions as variation around mean
            cell_props = np.random.dirichlet(homogeneity_alpha * safe_proportions)
            cell_proportions_list.append(cell_props)
        
        return np.array(cell_proportions_list)
    
    def get_simulation_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a summary report of the simulation
        """
        metadata = results['simulation_metadata']
        cluster_stats = results['cluster_stats']
        
        summary_lines = [
            "REALISTIC DATA SIMULATION SUMMARY",
            "=" * 50,
            "",
            "WORKFLOW: variants → clustering → signatures → cells",
            "",
            "Data Dimensions:",
            f"  - Mitochondrial positions: {metadata['n_locations']}",
            f"  - Variant molecules: {metadata['n_variants']}",
            f"  - Discovered clusters: {cluster_stats['n_clusters']}",
            f"  - Simulated cells: {metadata['n_cells']}",
            "",
            "Clustering Results:",
            f"  - Method: {metadata['clustering_method']}",
            f"  - Mean variants per cluster: {cluster_stats['mean_variants_per_cluster']:.2f}",
            f"  - Std variants per cluster: {cluster_stats['std_variants_per_cluster']:.2f}",
            f"  - Cluster sizes: {cluster_stats['cluster_sizes'].tolist()}",
            "",
            "Simulation Parameters:",
            f"  - Homogeneity alpha: {metadata['homogeneity_alpha']}",
            f"  - Noise level: {metadata['noise_level']}",
            f"  - Random seed: {metadata['random_state']}",
            "",
            "Output Data Shapes:",
            f"  - Cluster signatures (K_true): {results['K_true'].shape}",
            f"  - Cell proportions (P_cells_true): {results['P_cells_true'].shape}",
            f"  - Observed data (C_observed): {results['C_observed'].shape}",
        ]
        
        return "\n".join(summary_lines)


def generate_homogeneous_data_realistic(n_clusters: int = 50,
                                       n_locations: int = 1650,
                                       n_cells: int = 100,
                                       n_variants: int = 1000,
                                       min_variants_per_molecule: int = 1,
                                       max_variants_per_molecule: int = 20,
                                       clustering_method: str = 'kmeans',
                                       homogeneity_alpha: float = 100.0,
                                       noise_level: float = 0.01,
                                       variant_frequency_shape: float = 2.0,
                                       random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function for generating realistic homogeneous cell population data
    
    This function follows the same workflow as the iMiGseq notebook:
    1. Generate random variants (like UMIs from VAULT)
    2. Apply K-means clustering to discover patterns
    3. Extract cluster signatures
    4. Generate cell population based on discovered patterns
    
    Parameters match the original simulation for easy replacement
    """
    
    simulator = RealisticDataSimulator(random_state=random_state)
    
    results = simulator.generate_realistic_data(
        n_locations=n_locations,
        n_variants=n_variants,
        n_clusters=n_clusters,
        n_cells=n_cells,
        min_variants_per_molecule=min_variants_per_molecule,
        max_variants_per_molecule=max_variants_per_molecule,
        clustering_method=clustering_method,
        homogeneity_alpha=homogeneity_alpha,
        noise_level=noise_level,
        variant_frequency_shape=variant_frequency_shape
    )
    
    return results