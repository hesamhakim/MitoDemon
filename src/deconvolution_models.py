import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy.optimize import nnls, minimize
import cvxpy as cp
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch

class DeconvolutionModels:
    """Collection of deconvolution models for benchmarking"""
    
    @staticmethod
    def nnls_deconvolve(K, C_observed):
        """Non-negative least squares - guaranteed non-negative"""
        n_cells = C_observed.shape[0]
        p_estimated = np.zeros((n_cells, K.shape[1]))
        
        for i in range(n_cells):
            coef, _ = nnls(K, C_observed[i])
            # Normalize to sum to 1
            if coef.sum() > 0:
                p_estimated[i] = coef / coef.sum()
            else:
                p_estimated[i] = coef
                
        return p_estimated
    
    @staticmethod
    def constrained_regression(K, C_observed):
        """Constrained regression with sum-to-one constraint"""
        n_cells, n_positions = C_observed.shape
        n_clusters = K.shape[1]
        p_estimated = np.zeros((n_cells, n_clusters))
        
        for i in range(n_cells):
            # Define optimization variable
            p = cp.Variable(n_clusters)
            
            # Define objective (least squares)
            objective = cp.Minimize(cp.sum_squares(K @ p - C_observed[i]))
            
            # Define constraints
            constraints = [
                p >= 0,  # Non-negativity
                cp.sum(p) == 1  # Sum to one
            ]
            
            # Solve
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS)
            
            if p.value is not None:
                p_estimated[i] = p.value
                
        return p_estimated
    
    @staticmethod
    def ensemble_deconvolve(K, C_observed, methods=['nnls', 'ridge', 'elasticnet']):
        """Ensemble of multiple methods"""
        predictions = []
        
        if 'nnls' in methods:
            predictions.append(DeconvolutionModels.nnls_deconvolve(K, C_observed))
            
        if 'ridge' in methods:
            ridge = Ridge(alpha=0.001, positive=True)
            p_ridge = np.zeros_like(predictions[0])
            for i in range(C_observed.shape[0]):
                ridge.fit(K, C_observed[i])
                coef = ridge.coef_
                if coef.sum() > 0:
                    p_ridge[i] = coef / coef.sum()
            predictions.append(p_ridge)
            
        # Average predictions
        return np.mean(predictions, axis=0)
    
    @staticmethod
    def bayesian_deconvolve(K, C_observed, dirichlet_alpha_prior: np.ndarray, n_samples: int = 1000):
        """
        Bayesian deconvolution with Dirichlet prior (new: using Pyro for variational inference)
        
        Parameters:
        -----------
        K : np.ndarray
            Signatures (n_positions x n_clusters)
        C_observed : np.ndarray
            Observed data (n_cells x n_positions)
        dirichlet_alpha_prior : np.ndarray
            Dirichlet concentration parameters (n_clusters,)
        n_samples : int
            Number of posterior samples
        
        Returns:
        --------
        np.ndarray
            Estimated proportions (n_cells x n_clusters)
        np.ndarray
            Uncertainty (std dev) (n_cells x n_clusters)
        """
        n_cells, n_positions = C_observed.shape
        n_clusters = K.shape[1]
        
        K_torch = torch.tensor(K, dtype=torch.float32)
        C_torch = torch.tensor(C_observed, dtype=torch.float32)
        alpha_prior = torch.tensor(dirichlet_alpha_prior, dtype=torch.float32)
        
        def model(c_obs):
            p = pyro.sample("p", dist.Dirichlet(alpha_prior))
            with pyro.plate("data", n_positions):
                pyro.sample("obs", dist.Normal((p @ K_torch.T), 0.01), obs=c_obs)
        
        def guide(c_obs):
            alpha_q = pyro.param("alpha_q", alpha_prior.clone(), constraint=dist.constraints.positive)
            pyro.sample("p", dist.Dirichlet(alpha_q))
        
        p_estimated = np.zeros((n_cells, n_clusters))
        p_uncertainty = np.zeros((n_cells, n_clusters))
        
        for i in range(n_cells):
            svi = SVI(model, guide, Adam({"lr": 0.01}), Trace_ELBO())
            for _ in range(500):
                svi.step(C_torch[i])
            
            # Sample from posterior
            posterior_samples = []
            for _ in range(n_samples):
                posterior_samples.append(guide(C_torch[i]).detach().numpy())
            
            samples = np.array(posterior_samples)
            p_estimated[i] = samples.mean(axis=0)
            p_uncertainty[i] = samples.std(axis=0)
        
        return p_estimated, p_uncertainty
