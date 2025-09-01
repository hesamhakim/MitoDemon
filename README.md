# MitoDemon Pipeline README

## Overview
MitoDemon is a computational pipeline for deconvolving mitochondrial heteroplasmy from single-cell sequencing data. It processes VAF (Variant Allele Frequency) profiles to estimate proportions of distinct mitochondrial clusters within cells, supporting both simulated and real VCF data from the VAULT pipeline. The pipeline is implemented in Python via three Jupyter notebooks and four helper modules, enabling modular, reproducible analysis.

Key features:
- UMI-aware VCF processing with quality filtering and INDEL encoding.
- Advanced variant clustering (e.g., spectral with PCA for sparsity).
- Realistic cell population simulation with noise models (Gaussian, Poisson, UMI bias).
- Deconvolution using multiple models (NNLS, Constrained, Ensemble, Bayesian with Dirichlet priors).
- Comprehensive evaluation with parameter sweeps, metrics (R², MSE, MAE, JSD), and visualizations.

This version updates the original pipeline (as described in MitoDemon Project Description.docx) by incorporating continuous VAFs, INDEL support, heterogeneous simulation, Bayesian methods with historical priors, and detailed reporting—improving accuracy (e.g., R² ~0.966) and handling of real data.

## Prerequisites
- **Python Environment**: Python 3.11+ with libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, cvxpy, pyro-ppl, torch, joblib, tqdm. Install via `pip install -r requirements.txt` (create one if needed: list the above).
- **Hardware**: CPU with multi-core support for parallel sweeps; GPU optional for Pyro acceleration.
- **Input Data**: VAULT-generated VCF files (e.g., all_snp_from_perfect_umi.vcf) in a data directory.
- **Directory Structure**:
  ```
  MitoDemon/
  ├── notebooks/
  │   ├── iMiGseq_VAULT_snp_vcf_output_standardization_05.ipynb
  │   ├── Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression_05.ipynb
  │   └── Priors_Extraction_and_Deconvolution_05.ipynb
  ├── src/
  │   ├── cell_simulator.py
  │   ├── deconvolution_models.py
  │   ├── variant_clustering.py
  │   └── vcf_processor.py
  ├── data/  # Place VCF files here (e.g., vault_pipeline_output/)
  └── sc_mito_vars/  # Output directory (auto-created)
  ```

Run in a Jupyter environment (e.g., JupyterLab or VS Code). Set sys.path to include '../src' if running notebooks.

## Workflow and How to Run
The pipeline consists of three notebooks run sequentially (recommended order: 1 → 2 → 3) for a full analysis. Each notebook is self-contained but builds on outputs from previous ones (e.g., .npz files). Total runtime ~10-30 min per notebook on standard hardware, depending on data size.

### Recommended Sequence
1. **Run Notebook 1: iMiGseq_VAULT_snp_vcf_output_standardization_05.ipynb** (Process VCF and Simulate Cells)
   - Processes a single VCF, clusters variants, simulates cell populations.
   - Command: Open in Jupyter and execute all cells (Ctrl+Shift+Enter).
   - Outputs: Timestamped directory (e.g., sc_mito_vars/real_data/imigseq_SRR12676843/) with .npz (K_true, P_cells_true, C_observed), .npy matrices, .csv tables, .json metadata/parameters, plots, and summary.txt.

2. **Run Notebook 2: Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression_05.ipynb** (Deconvolve and Evaluate)
   - Loads data from Notebook 1, runs deconvolution models with sweeps, generates reports/visualizations.
   - Update `sample` in Cell 2 to match Notebook 1's output (e.g., "SRR12676843").
   - Command: Execute all cells.
   - Outputs: deconvolution_report.txt (ranked models, metrics), model_comparison_results.csv, plots (e.g., proportion traces, residuals), validation stats.

3. **Run Notebook 3: Priors_Extraction_and_Deconvolution_05.ipynb** (Extract Priors and Bayesian Deconvolution)
   - Processes multiple historical VCFs to fit priors, applies Bayesian model to data from Notebook 1.
   - Update vcf_paths in Cell 2 if needed.
   - Command: Execute all cells.
   - Outputs: dirichlet_prior.npy, printed alphas/uncertainties, R² metrics.

**Tips for Running**:
- Set RANDOM_SEED for reproducibility.
- For large VCFs, increase PCA_COMPONENTS or use GPU for Pyro in Notebook 3.
- If errors occur (e.g., dimension mismatch), check N_MITO_POSITIONS consistency across notebooks (default 16299 for mouse; adjust for other organisms).
- Parallel execution: Sweeps in Notebook 2 use joblib; enable multi-core with n_jobs=-1.

## All Parameters
Parameters are defined in notebook cells (e.g., Cell 2 configs) and modules. Below is a comprehensive list grouped by category, with defaults, descriptions, and impacts (from outputs and code).

### General Parameters (All Notebooks)
- **RANDOM_SEED** (default: 42): Seed for reproducibility; affects random processes like clustering/simulation. Impact: Ensures consistent results across runs.
- **OUTPUT_DIR** (default: "../sc_mito_vars/real_data"): Base output path; auto-creates subdirs.

### VCF Processing Parameters (Notebook 1, vcf_processor.py)
- **FILTER_PASS** (default: True): Keep only PASS variants. Impact: Reduces noise but may discard data if False.
- **INCLUDE_INDELS** (default: True): Include INDELs (encoded as presence/length/type features). Impact: Expands matrix (e.g., +140,223 features), improves realism but increases computation.
- **MIN_VAF** (default: 0.25): Min allele frequency. Impact: Filters low-confidence variants; lower values retain more but increase noise.
- **MIN_DEPTH** (default: 1): Min total reads. Impact: Ensures reliability; very low for relaxed filtering in sparse data.
- **MIN_ALT_READS** (default: 1): Min alternate reads. Impact: Similar to depth; minimal for inclusivity.
- **MAX_MISSING_RATE** (default: 0.95): Max missing UMIs per variant (retains if in >=5% UMIs). Impact: Relaxes to keep rare but recurrent variants.
- **N_MITO_POSITIONS** (default: 16569, human; update to 16299 for mouse): mtDNA length for matrix rows. Impact: Critical for position mapping; mismatch causes slicing errors.

### Clustering Parameters (Notebook 1/3, variant_clustering.py)
- **N_CLUSTERS** (default: 20): Number of mitochondrial groups. Impact: Higher increases resolution but risks overfitting in sparse data.
- **CLUSTERING_METHOD** (default: 'spectral'): Method ('kmeans', 'nmf', etc.). Impact: Spectral handles sparsity best (RBF affinity avoids NaNs).
- **USE_PCA** (default: True): Apply PCA pre-clustering. Impact: Reduces dims (e.g., to 50) for high-sparsity matrices (>0.999).
- **PCA_COMPONENTS** (default: 50): PCA output dims. Impact: Balances info loss vs computation; auto-caps at min(samples, features-1).
- **CONTINUOUS** (default: True): Use VAF values (vs binary). Impact: Preserves heteroplasmy levels for accurate signatures.

### Simulation Parameters (Notebook 1, cell_simulator.py)
- **N_CELLS** (default: 200): Cells to simulate. Impact: Larger populations for robust stats; scales computation.
- **HOMOGENEITY_ALPHA** (default: 50.0): Dirichlet alpha for variation. Impact: Higher = more homogeneous proportions.
- **SKEW_FACTOR** (default: 2.0): Skew towards low-mutated clusters. Impact: Models dominance of wild-type mitochondria.
- **NOISE_LEVEL** (default: 0.01): Gaussian SD. Impact: Simulates sequencing noise; higher degrades signal.
- **DROPOUT_RATE** (default: 0.1): Zero probability. Impact: Models dropouts in sc-data.
- **OVERDISPERSION** (default: 0.05): Beta variance. Impact: Adds VAF variability; 0 disables.
- **POISSON_LAMBDA** (default: 10.0): Poisson mean for counts. Impact: Models read variability; higher = less noise.
- **UMI_BIAS_RATE** (default: 0.1): UMI amplification bias. Impact: Simulates technical artifacts.

### Deconvolution Parameters (Notebook 2, deconvolution_models.py)
- Sweep params (Cell 3): include_indels (True/False), noise_types (gaussian/poisson), use_priors (True/False). Impact: Tests 8 configs for robustness.
- **n_samples** (default: 1000, Bayesian): Posterior samples. Impact: More = better uncertainty estimates but slower.
- GridSearchCV in models (e.g., ElasticNet alpha/l1_ratio). Impact: Auto-tunes regularization.

### Priors Extraction Parameters (Notebook 3)
- **vcf_paths**: List of historical VCFs. Impact: More files = better prior fitting.
- **n_steps** (default: 1000, SVI): Optimization steps. Impact: Higher for convergence.
- **lr** (default: 0.01, Adam): Learning rate. Impact: Balances speed/stability.

## How to Run the Notebooks
1. **Setup**: Clone repo, install prerequisites, place VCFs in data/.
2. **Run Sequence**: 1 (process/simulate) → 2 (deconvolve/evaluate) → 3 (priors/Bayesian).
3. **Customization**: Edit Cell 2 params in each notebook (e.g., change N_CLUSTERS to 10 for fewer groups).
4. **Output Location**: Check sc_mito_vars/ for results.

## Interpreting the Data
- **Outputs from Notebook 1**: variant_matrix.npy (features x UMIs, sparsity ~0.9999 indicates mostly zeros—typical for mtDNA). cluster_signatures (positions x clusters, mean VAF per position). P_cells_true (cells x clusters, sums to 1 per cell). C_observed (cells x positions, noisy VAFs). Visuals show cluster sizes (often skewed), proportions (e.g., one dominant cluster ~0.7247). Metadata.json has stats like mean_variants_per_cluster (~0.00 due to sparsity).
- **Outputs from Notebook 2**: deconvolution_report.txt ranks models (e.g., Constrained best with R²=0.9658, inf JSD from negatives—clip in vis). model_comparison_results.csv for metrics. Plots: Proportion traces (true vs estimated, R²=0.9658 shows good fit), residuals (~0 mean, narrow dist), per-cluster R² (varies 0.07-0.63), error hist (symmetric around 0). Validation: Sums ~1.0, sparsity 0.044 (true) vs 0.000 (estimated—model fills zeros).
- **Outputs from Notebook 3**: dirichlet_prior.npy (alphas e.g., [4.099, ...] favor sparse clusters). p_estimated (cells x clusters, R²=0.9665 vs true). Uncertainty (low mean 0.0075 indicates confidence). Interpret priors as concentration (higher = more mass on that cluster); low uncertainty suggests stable estimates.

For questions, refer to module docstrings or project description. This pipeline is designed for iterative use—experiment with params to match your data.