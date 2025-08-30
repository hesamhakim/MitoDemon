"""
Cell Simulation Module for Single-Cell Mitochondrial Analysis
Author: [Your Name]
Date: 2025
Description: Simulates additional cells based on observed single-cell data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

class CellSimulator:
    """Simulate additional cells based on observed single-cell data"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize simulator
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def simulate_cell_proportions(self,
                                 observed_proportions: np.ndarray,
                                 n_cells: int = 100,
                                 homogeneity_alpha: float = 100.0,
                                 skew_factor: float = 2.0) -> np.ndarray:
        """
        Simulate cell proportions for multiple cells
        
        Parameters:
        -----------
        observed_proportions : np.ndarray
            Observed proportions from single cell (1 x n_clusters)
        n_cells : int
            Number of cells to simulate
        homogeneity_alpha : float
            Controls variation between cells (higher = more homogeneous)
        skew_factor : float
            Controls skew in proportions
        
        Returns:
        --------
        np.ndarray
            Simulated cell proportions (n_cells x n_clusters)
        """
        n_clusters = observed_proportions.shape[-1]
        
        # Use observed proportions as base
        base_proportions = observed_proportions.flatten()
        
        # Add small pseudocount to avoid zeros
        base_proportions = np.where(base_proportions == 0, 1e-6, base_proportions)
        
        # Normalize
        base_proportions = base_proportions / base_proportions.sum()
        
        # Apply skew factor
        skewed_proportions = base_proportions ** skew_factor
        skewed_proportions = skewed_proportions / skewed_proportions.sum()
        
        # Generate cell proportions using Dirichlet distribution
        cell_proportions = []
        
        for _ in range(n_cells):
            # Sample from Dirichlet with concentration parameter
            alpha_params = homogeneity_alpha * skewed_proportions
            cell_props = np.random.dirichlet(alpha_params)
            cell_proportions.append(cell_props)
        
        return np.array(cell_proportions)
    
    def add_noise(self,
                  cell_proportions: np.ndarray,
                  cluster_signatures: np.ndarray,
                  noise_level: float = 0.01) -> np.ndarray:
        """
        Add noise to create observed data
        
        Parameters:
        -----------
        cell_proportions : np.ndarray
            Cell proportion matrix (n_cells x n_clusters)
        cluster_signatures : np.ndarray
            Cluster signatures (n_positions x n_clusters)
        noise_level : float
            Standard deviation of Gaussian noise
        
        Returns:
        --------
        np.ndarray
            Observed data with noise (n_cells x n_positions)
        """
        # Matrix multiplication: (n_cells x n_clusters) @ (n_clusters x n_positions)
        C_true = cell_proportions @ cluster_signatures.T
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, size=C_true.shape)
        C_observed = C_true + noise
        
        # Clip to [0, 1] range
        C_observed = np.clip(C_observed, 0, 1)
        
        return C_observed
    
    def simulate_heterogeneous_population(self,
                                         observed_proportions: np.ndarray,
                                         n_cells: int = 100,
                                         n_subpopulations: int = 3,
                                         subpop_weights: Optional[np.ndarray] = None,
                                         homogeneity_alpha: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate heterogeneous cell population with multiple subpopulations
        
        Parameters:
        -----------
        observed_proportions : np.ndarray
            Observed proportions from single cell
        n_cells : int
            Total number of cells to simulate
        n_subpopulations : int
            Number of distinct subpopulations
        subpop_weights : Optional[np.ndarray]
            Weights for each subpopulation (default: equal)
        homogeneity_alpha : float
            Within-subpopulation homogeneity
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            cell_proportions: Simulated proportions (n_cells x n_clusters)
            subpop_labels: Subpopulation label for each cell
        """
        n_clusters = observed_proportions.shape[-1]
        
        if subpop_weights is None:
            subpop_weights = np.ones(n_subpopulations) / n_subpopulations
        
        # Normalize weights
        subpop_weights = subpop_weights / subpop_weights.sum()
        
        # Determine number of cells per subpopulation
        cells_per_subpop = np.random.multinomial(n_cells, subpop_weights)
        
        # Create distinct subpopulation centers
        subpop_centers = []
        base_props = observed_proportions.flatten()
        
        for i in range(n_subpopulations):
            # Perturb base proportions for each subpopulation
            perturbation = np.random.dirichlet(np.ones(n_clusters) * 10)
            subpop_center = 0.7 * base_props + 0.3 * perturbation
            subpop_center = subpop_center / subpop_center.sum()
            subpop_centers.append(subpop_center)
        
        # Generate cells for each subpopulation
        all_cell_proportions = []
        all_subpop_labels = []
        
        for subpop_idx, (n_subpop_cells, subpop_center) in enumerate(zip(cells_per_subpop, subpop_centers)):
            if n_subpop_cells > 0:
                # Generate cells around this subpopulation center
                subpop_props = self.simulate_cell_proportions(
                    subpop_center.reshape(1, -1),
                    n_cells=n_subpop_cells,
                    homogeneity_alpha=homogeneity_alpha
                )
                all_cell_proportions.append(subpop_props)
                all_subpop_labels.extend([subpop_idx] * n_subpop_cells)
        
        cell_proportions = np.vstack(all_cell_proportions)
        subpop_labels = np.array(all_subpop_labels)
        
        # Shuffle to mix subpopulations
        shuffle_idx = np.random.permutation(n_cells)
        cell_proportions = cell_proportions[shuffle_idx]
        subpop_labels = subpop_labels[shuffle_idx]
        
        return cell_proportions, subpop_labels
