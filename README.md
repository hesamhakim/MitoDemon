# MitoDemon: A Hierarchical Bayesian Model for Mitochondrial Heteroplasmy

## 1. Objective

This repository contains a complete Python-based pipeline for simulating and modeling mitochondrial heteroplasmy. The primary goal is to accurately estimate the proportions of different mitochondrial clusters within a population of cells by leveraging multi-level data.

The project uses a **three-level hierarchical Bayesian model** implemented in **PyMC**. This sophisticated approach allows us to integrate information simultaneously from:
* **Tissue-level** VAF (Variant Allele Frequency) data
* **Single-cell-level** VAF data
* The known **cell clone lineage tree**
* The grouping of **cells within each clone**

By building a model that mirrors the biological processes of clonal evolution and intra-clone variation, we can achieve a more robust and accurate deconvolution of cellular composition. This pipeline serves as a proof-of-concept, demonstrating the methodology on a realistic, simulated dataset before it is applied to real experimental data.

---

## 2. Pipeline Overview

The pipeline is divided into two main parts, each contained in a separate Jupyter Notebook:

1.  **`01_data_simulation.ipynb`**: Generates a synthetic dataset with a known ground truth. This includes a lineage of cell clones and the individual cells that make up each clone.
2.  **`02_model_implementation.ipynb`**: Implements the Bayesian model to estimate the hidden parameters from the simulated data and validates the results.

---

## 3. Data Structure and Parameters

The entire simulated dataset is stored in a single compressed file: `simulated_data.npz`.

### Data Parameters (Configurable in `01_data_simulation.ipynb`)
* `N_CLUSTERS`: The total number of distinct mitochondrial clusters.
* `N_LOCATIONS`: The number of mitochondrial genomic locations (variant sites) being tracked.
* `N_CLONES`: The target number of cell clones to simulate.
* `N_GENERATIONS`: The number of evolutionary steps for clone generation, which controls the depth of the lineage tree.
* `MIN/MAX_CELLS_PER_CLONE`: The range for the number of individual cells to simulate within each clone.
* `MIN/MAX_VARIANTS_PER_CLUSTER`: The range for the number of signature variants that define each mitochondrial cluster.

### Data Structure (Contents of `simulated_data.npz`)
* `K_true`: A `(N_LOCATIONS, N_CLUSTERS)` matrix representing the **known VAF signatures**. Based on the latest assumption, this is a **binary matrix** where a `1` indicates a mutation is present and `0` indicates it is not.
* `P_clones_true`: A `(N_CLONES, N_CLUSTERS)` matrix holding the **ground truth proportion** of each cluster within every **clone**.
* `P_cells_true`: A `(N_CELLS, N_CLUSTERS)` matrix holding the **ground truth proportion** of each cluster within every individual **cell**.
* `clone_lineage_map`: A Python dictionary defining the mother-daughter relationships for the **clones**.
* `cell_to_clone_map`: A Python dictionary mapping each individual cell ID to its parent clone ID.
* `C_observed`: A `(N_CELLS, N_LOCATIONS)` matrix of the **"observed" single-cell VAFs**.
* `VAF_tissue_observed`: A `(N_LOCATIONS,)` array representing the **"observed" tissue VAF profile**.

---

## 4. Methodology

### Simulation Strategy (`01_data_simulation.ipynb`)
The synthetic data is generated to realistically mimic biological processes:
1.  **Define Signatures**: Create distinct, sparse, and **binary (0/1)** VAF signatures for each of the `N_CLUSTERS`.
2.  **Simulate Clone Lineage**: A root clone is assigned initial proportions. The simulation then iteratively generates daughter clones for `N_GENERATIONS`. The proportions of each daughter clone are drawn from a **Dirichlet distribution** centered on its mother's proportions, simulating clonal evolution and drift.
3.  **Simulate Cells within Clones**: For each clone, a small population of individual cells is generated. The proportions for each cell are drawn from a tight Dirichlet distribution centered on its parent clone's proportions, simulating intra-clone heterogeneity.
4.  **Generate Observations**: The "observed" VAF data is created by mixing the binary cluster signatures according to the true cell proportions (`P_cells_true`) and adding a small amount of Gaussian noise to simulate experimental measurement error.

### Model Implementation Strategy (`02_model_implementation.ipynb`)
The three-level hierarchical Bayesian model is constructed in PyMC:
1.  **Priors**: We define priors for the model's global parameters, including the drift rates (`alpha_inter_clone`, `alpha_intra_clone`) and the observational noise (`sigma`).
2.  **Clone Hierarchy**: The proportions for each daughter clone are drawn from a `Dirichlet` prior centered on its mother's estimated proportions.
3.  **Cell Hierarchy**: The proportions for each individual cell are drawn from a `Dirichlet` prior centered on its parent clone's estimated proportions.
4.  **Likelihoods**: The model is constrained by the observed data. The estimated cell proportions (`P_cells_est`) are used to generate the expected VAFs, which are then compared against the `C_observed` and `VAF_tissue_observed` data.
5.  **Inference**: The model uses the **NUTS sampler (MCMC)** to generate samples from the posterior distribution of all unknown parameters. Validation is performed by comparing these estimates to the ground truth.

---

## 5. How to Run

1.  **Run the Simulation Notebook**: Execute all cells in `01_data_simulation.ipynb`. This will create the `simulated_data.npz` file.
2.  **Run the Model Notebook**: Execute all cells in `02_model_implementation.ipynb`. This will load the data file and run the MCMC sampling to produce results and validation metrics.

---

## 6. File Descriptions

* **`01_data_simulation.ipynb`**: Notebook for generating and saving the synthetic dataset.
* **`02_model_implementation.ipynb`**: Notebook for building, running, and validating the PyMC model.
* **`simulated_data.npz`**: Compressed file containing the dictionary of all simulated data arrays.
* **`model_results_3level.nc`**: Saved output of the MCMC sampling process, allowing for re-analysis without re-running the sampler.