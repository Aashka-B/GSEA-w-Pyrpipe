# Reproducibility Study of Pancreatic Cancer lncRNAs and Machine Learning Prediction

## Overview

This repository is dedicated to a reproducibility study of the original research that investigated the role of long noncoding RNAs (lncRNAs) as potential biomarkers for metastatic progression in pancreatic cancer using machine learning techniques. The aim of this reproducibility study is to validate the findings of the original research, ensuring that the results are consistent and can be reliably used for further scientific exploration and clinical application.

## Original Study

### Title

*Machine Learning Predicts Metastatic Progression using Novel Differentially Expressed lncRNAs as Potential Markers in Pancreatic Cancer*

### Goals

The original study sought to:
- Identify novel lncRNAs significantly altered in pancreatic cancer.
- Utilize machine learning techniques to predict metastatic cases of pancreatic cancer based on identified lncRNAs.
- Perform differential gene expression analysis (DGEA) and gene set enrichment analysis (GSEA) to understand the functions of these lncRNAs.
- Investigate the potential of these lncRNAs as biomarkers for metastatic progression, providing insights for diagnosis and treatment strategies.

### Data Source

The Cancer Genome Atlas (TCGA) served as the primary data source for RNA-sequencing transcriptomic profiles of pancreatic carcinomas.

## Programs Used

The reproducibility study utilized two Python scripts originally developed for the analysis:

- `lncDGEA_with_GSEA.py`: Handles data preprocessing, DGEA, GSEA, and prepares the data for machine learning analysis.
- `ml.py`: Focuses on applying and evaluating various machine learning models including Logistic Regression, Support Vector Machine (SVM), Random Forest Classifier, and eXtreme Gradient Boosting Classifier to predict metastatic progression.

## Environment Requirements

The study was conducted in a specific computational environment to ensure the reproducibility of results. Key requirements include:

- Python 3.8.18
- Libraries such as NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, GSEAPy, PyDESeq2, and others as listed in the `PanC.yml` file.

A Conda environment file, `PanC.yml`, is provided to recreate the exact computational environment used in this study.

## Setting Up the Environment

To recreate the environment:

```bash
conda env create -f PanC.yml
