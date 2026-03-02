# CSE 284 Project - GWAS vs Linear Mixed Models: A Stress-Test Comparison Using 1000 Genomes chr22

**Contributers:**

Astoria Ma (PID: A16913550), Chloe Huang (PID: A69042803)

# Project Overview

This project evaluates the statistical calibration and robustness of:

- Standard GWAS (PLINK regression)
- Linear Mixed Models (GEMMA LMM)

under controlled stress scenarios using 1000 Genomes chromosome 22 data.

We construct genome-wide covariates (chr20–22) for population structure and relatedness, and perform association testing on chr22.

The goal is to systematically test when LMM provides benefits over standard regression — and when it over- or under-corrects.


## Data Preparation


### Association dataset

All raw data files are downloaded from the datahub and then saved to `data/` folder. Preprocessed data are saved to `data_preprocessed/` folder. 

Used for:
- GWAS regression (PLINK)
- LMM testing (GEMMA -lmm 4)

### Genome-wide dataset (chr20–22)

Used for:
- PCA (population structure)
- GRM construction (relatedness modeling)

Steps:
1. VCF → PLINK conversion
2. QC filtering (`--maf`, `--geno`, `--hwe`)
3. LD pruning
4. PCA computation
5. GRM construction using GEMMA (`-gk 1`)


# Stress Scenarios

We evaluate four stress conditions:

### S1 — Population Structure Confounding
Inject phenotype signal aligned with PC1.

### S2 — Case-Control Imbalance
Simulate binary traits with varying prevalence.

### S3 — Sample Relatedness
Compare high-related vs low-related cohorts.

### S4 — Genotype Missingness
Randomly mask genotypes at controlled rates.


