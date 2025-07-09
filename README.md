# MitoDemon: A Framework for Mitochondrial Heteroplasmy Deconvolution

## 1\. Objective

This repository contains a complete Python-based pipeline for simulating and modeling mitochondrial heteroplasmy. The project provides two distinct methodologies to estimate the proportions of different mitochondrial clusters within a population of cells, tailored to different biological assumptions and data structures.

The two main approaches are:

1.  A **Hierarchical Bayesian Model** designed for complex, multi-level data involving a lineage of distinct cell clones.
2.  A **Regularized Linear Regression Model** designed for the simpler case of a single, homogeneous population of cells.

This repository provides the code to simulate data for both scenarios and to implement the corresponding deconvolution models, allowing for robust validation and analysis.

-----

## 2\. Methodologies & Workflows

This project is organized into two primary workflows, each with its own simulation and implementation notebooks.

### Methodology 1: Hierarchical Bayesian Model for Complex Lineages

This approach is designed for the most complex-case scenario where the cell population is composed of multiple, distinct **clones** that evolve over time, and each clone contains a population of individual **cells**.

  * **Model:** A three-level hierarchical Bayesian model implemented in PyMC.
  * **Use Case:** Ideal for detailed, multi-level experimental data that includes a known or inferred cell clone lineage. It can deconvolve proportions at both the clone and single-cell level while accounting for evolutionary drift.
  * **Notebooks:**
      * **Simulation:** `notebooks/Data_Simulation_Muti_Clone_Multi_leve_Mitochondria.ipynb`
      * **Implementation:** `notebooks/Deconvolution_Hierarchial_Bayesian_Model_Implementation.ipynb`

### Methodology 2: Regularized Linear Regression for Homogeneous Populations

This approach is a faster, more direct method designed for the common scenario where the data comes from a single, relatively homogeneous population of cells.

  * **Model:** A Regularized Linear Regression model (`ElasticNet`) implemented in scikit-learn.
  * **Use Case:** Perfect for deconvolving cluster proportions from a set of single-cell VAFs when there is no known lineage information and the cells are assumed to be minor variations of a single mean composition.
  * **Notebooks:**
      * **Simulation:** `notebooks/Data_Simulation_Homogeneous_Cell_Population.ipynb`
      * **Implementation:** `notebooks/Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression.ipynb`

-----

## 3\. How to Run

1.  **Choose a Methodology**: Decide which of the two workflows matches your research question and data structure.
2.  **Run the Simulation Notebook**: Execute the appropriate simulation notebook (e.g., `Data_Simulation_Homogeneous_Cell_Population.ipynb`) to generate a synthetic dataset. This will create a `.npz` file in the `sim_data` directory.
3.  **Run the Implementation Notebook**: Execute the corresponding implementation notebook (e.g., `Deconvolution_Homogenous_Cell_by_Regularized_Linear_Regression.ipynb`). This will load the simulated data and run the deconvolution model to produce results and validation metrics. The results of the model fitting will be saved in the `model` directory.

-----

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