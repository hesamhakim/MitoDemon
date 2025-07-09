# MitoDemon: A Framework for Mitochondrial Heteroplasmy Deconvolution

## 1\. Objective

This repository contains a complete Python-based pipeline for simulating and modeling mitochondrial heteroplasmy. The project provides two distinct, powerful methodologies to estimate the proportions of different mitochondrial clusters within a population of cells, each tailored to different biological assumptions and data structures.

The goal is to provide a robust framework for researchers to deconvolve complex single-cell VAF data, supported by a simulation engine that allows for rigorous model validation.

## 2\. Methodologies & Workflows

This project is organized into two primary workflows.

### **Methodology 1: Hierarchical Bayesian Model for Complex Lineages**

This is a sophisticated, multi-level model designed for the most complex-case scenario where the cell population is composed of multiple, distinct **clones** that evolve over time, and each clone contains a population of individual **cells**.

  * **Objective:** To accurately deconvolve cluster proportions at both the clone and single-cell level by leveraging the known lineage structure. This model can simultaneously estimate the rate of mitochondrial drift between and within clones.
  * **Assumptions:**
      * The cell population is heterogeneous and composed of multiple distinct clones.
      * The lineage tree of the clones is known.
      * The mapping of each individual cell to its parent clone is known.
      * The VAF signatures of the underlying mitochondrial clusters are known.
  * **Input Data:**
      * `K_true`: A binary matrix of known mitochondrial cluster signatures.
      * `C_observed`: The observed VAF for each individual cell.
      * `VAF_tissue_observed`: The observed VAF for the bulk tissue sample.
      * `clone_lineage_map`: A dictionary defining the mother-daughter relationships for the clones.
      * `cell_to_clone_map`: A dictionary mapping each cell to its parent clone.
  * **Output Parameters Estimated by the Model:**
      * `P_clones_est`: The proportion of each mitochondrial cluster for every **clone**.
      * `P_cells_est`: The proportion of each mitochondrial cluster for every **cell**.
      * `alpha_inter_clone`: The drift rate (concentration parameter) between mother and daughter clones.
      * `alpha_intra_clone`: The drift rate (concentration parameter) between a parent clone and its constituent cells.
      * `sigma_cell` / `sigma_tissue`: The observational noise terms.
  * **Notebooks:**
      * **Simulation:** `notebooks/Data_Simulation_Muti_Clone_Multi_leve_Mitochondria.ipynb`
      * **Implementation:** `notebooks/Deconvolution_Hierarchial_Bayesian_Model_Implementation.ipynb`

### **Methodology 2: Regularized Linear Regression for Homogeneous Populations**

This is a faster, more direct method designed for the common scenario where the data comes from a single, relatively homogeneous population of cells.

  * **Objective:** To provide a rapid and accurate deconvolution of cluster proportions for a set of single cells that are assumed to be minor variations of a single mean composition.
  * **Assumptions:**
      * The cell population is homogeneous (originating from a single clone).
      * There is no known lineage information connecting the cells.
      * The VAF signatures of the underlying mitochondrial clusters are known.
  * **Input Data:**
      * `K_true`: A binary matrix of known mitochondrial cluster signatures.
      * `C_observed`: The observed VAF for each individual cell.
  * **Output Parameters Estimated by the Model:**
      * `p_cells_estimated`: The proportion of each mitochondrial cluster for every **cell**.
  * **Notebooks:**
      * **Simulation:** `notebooks/Data_Simulation_Homogeneous_Cell_Population.ipynb`
      * **Implementation:** `notebooks/Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression.ipynb`

## 3\. How to Run

1.  **Choose a Methodology**: Decide which of the two workflows matches your research question and data structure.
2.  **Run the Simulation Notebook**: Execute the appropriate simulation notebook (e.g., `Data_Simulation_Homogeneous_Cell_Population.ipynb`) to generate a synthetic dataset. This will create a `.npz` file in the `sim_data` directory.
3.  **Run the Implementation Notebook**: Execute the corresponding implementation notebook (e.g., `Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression.ipynb`). This will load the simulated data and run the deconvolution model to produce results and validation metrics. The results of the model fitting will be saved in the `model` directory.

## 4\. Repository Structure

The repository is organized into three main directories:

```
.
├── notebooks/
│   ├── Data_Simulation_Muti_Clone_Multi_leve_Mitochondria.ipynb
│   ├── Deconvolution_Hierarchial_Bayesian_Model_Implementation.ipynb
│   ├── Data_Simulation_Homogeneous_Cell_Population.ipynb
│   └── Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression.ipynb
│
├── sim_data/
│   ├── simulated_data_hierarchical.npz
│   └── simulated_data_regression.npz
│
└── model/
    └── model_results_3level.nc
```

  * **`notebooks/`**: Contains the Jupyter notebooks for data simulation and model implementation for both methodologies.
  * **`sim_data/`**: The default location for the output of the simulation notebooks.
  * **`model/`**: The default location for the saved output of the MCMC sampling process from the Bayesian model.