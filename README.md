Of course. Here is the updated `README.md` file in markdown format.

# MitoDemon: A Framework for Mitochondrial Heteroplasmy Deconvolution

## 1\. Objective

This repository contains a complete Python-based pipeline for simulating, processing, and deconvolving mitochondrial heteroplasmy data from single-cell sequencing. The project implements a robust, regularized linear regression model (**ElasticNet**) to estimate the proportions of different mitochondrial clusters within a population of cells.

The framework provides a seamless workflow, from generating synthetic data for validation to processing real-world, UMI-tagged VCF files from the VAULT pipeline. The ultimate goal is to offer a comprehensive toolkit for researchers to explore mitochondrial dynamics, supported by a powerful simulation and validation engine.

## 2\. Main Workflow & Notebooks

The primary workflow of this project is organized into two main branches: a simulation branch for model development and robustness testing, and an experimental branch for analyzing real-world data.

### **Branch 1: Simulation and Deconvolution**

This workflow is designed to test the deconvolution model's accuracy and sensitivity under a wide range of controlled, synthetic conditions.

  * **Notebook:** `Simulation_and_Deconvolution_Sweep.ipynb`
      * **Functionality:** Generates synthetic single-cell VAF data with a variety of configurable parameters. It then runs the `ElasticNet` deconvolution model and performs a systematic parameter sweep to assess how each parameter affects the model's estimation accuracy.
      * **Key Simulation Parameters:**
          * `NOISE_LEVEL`: Simulates sequencing error.
          * `HOMOGENEITY_ALPHA`: Controls cell-to-cell variation.
          * `SKEW_FACTOR`: Biases cluster abundance based on mutation load.
          * `MUTATION_DENSITY_SHAPE`: Controls the distribution of mutation counts per cluster.
          * `PROPORTION_SKEW`: Controls the distribution of cluster sizes.
      * **Output:** A comprehensive numerical report and a series of visualizations detailing the model's performance under different simulated conditions.

### **Branch 2: Experimental Data Processing and Validation**

This workflow is designed for processing real-world data and validating the deconvolution model against a "ground truth" derived from high-resolution sequencing.

  * **Notebook 1 (Data Prep):** `Data_Preparation_Kadam.ipynb`
      * **Functionality:** This notebook processes pre-existing CSV data from Kadam et al. (2023), "Single-mitochondrion sequencing uncovers distinct mutational patterns and heteroplasmy landscape in mouse astrocytes and neurons." It merges inherited and somatic variant tables, transforms the data into a tidy long format, and adds useful annotations like the number of variants and mitochondria per cell.
  * **Notebook 2 (Alternative Data Prep):** `iMiGseq_VAULT_snp_vcf_output_standardization.html`
      * **Functionality:** A powerful, end-to-end pipeline that processes raw, UMI-tagged VCF files from the VAULT pipeline. It filters variants, clusters mitochondrial molecules (UMIs) to generate variant signatures, and simulates a cell population.
      * **Key Feature:** This notebook bridges the gap between raw sequencing data and the deconvolution model, creating an output that is directly compatible with the validation notebook. It uses a modular design with scripts in the `src/` directory for processing, clustering, and simulation.
      * **Key Filtering Parameters:** `MIN_VAF`, `MIN_DEPTH`, `MIN_ALT_READS`.
  * **Notebook 3 (Validation):** `Experimental_Validation_Sweep.ipynb`
      * **Functionality:** This notebook serves as the validation engine for the experimental data. It takes the output from either of the data preparation notebooks, clusters the individual mitochondria to create a ground truth, and then tests the deconvolution model's ability to rediscover this structure from a bulk signal.
      * **Key Feature:** Includes a parameter sweep to test how different clustering methods (e.g., `cosine` vs. `euclidean` distance) and signature types (binary vs. continuous) affect the model's accuracy.

## 3\. How to Run

1.  **Set Up Environment**: Ensure you have the necessary Python libraries installed (pandas, numpy, scikit-learn, seaborn, etc.).
2.  **Choose a Workflow**:
      * **For robustness testing on synthetic data**:
        1.  Open and run the `Simulation_and_Deconvolution_Sweep.ipynb` notebook.
        2.  Adjust the parameter ranges in the first cell to define your experiments.
        3.  The notebook will generate data, run the model, and output the results.
      * **For analyzing real data**:
        1.  **Prepare the data**: Run either `Data_Preparation_Kadam.ipynb` (for the Kadam CSVs) or the `iMiGseq_VAULT_snp_vcf_output_standardization.html` (for your own VCF files).
        2.  **Run validation**: Open the `Experimental_Validation_Sweep.ipynb` notebook. Ensure the `INPUT_PATH` in the first cell points to the output of your data preparation step.
        3.  Adjust the parameter sweep settings (e.g., `K_VALUES_TO_TEST`) as needed.
        4.  Execute the notebook to run the validation.
3.  **Review Outputs**: Check the `sim_data/` and `model/` directories for the generated files, including `.npy` data, `.csv` reports, and metadata files.

## 4\. Repository Structure

The repository is organized into several key directories:

```
.
├── notebooks/
│   ├── Simulation_and_Deconvolution_Sweep.ipynb
│   ├── Data_Preparation_Kadam.ipynb
│   ├── Experimental_Validation_Sweep.ipynb
│   └── (Additional VCF and experimental notebooks)
│
├── sc_mito_vars (Output from simulation and data processing notebooks)
│   └── real_data
│      └── imigseq_SRR12455630_run01
│   └── simdata_data
│      └── sim_run01
│
├── model/
│   └── (Saved model outputs and results)
│
└── src/
    └── (Python modules for VCF processing, clustering, etc.)
```

  * **`notebooks/`**: Contains all the Jupyter notebooks for the primary simulation and experimental workflows.
  * **`sim_data/`**: The default location for the output of the simulation and data processing notebooks. Each run should create a unique subdirectory containing data (`.npy`, `.csv`) and metadata (`.json`) files.
  * **`model/`**: The default location for saved model outputs, such as the results of the parameter sweeps.
  * **`src/`**: Contains the core Python modules used by the VCF processing notebook.