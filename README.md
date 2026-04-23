# Deeplearning2
Deeplearning project


# Cardiac Disease Classification & MRI Segmentation (ACDC)

This project focuses on the automatic segmentation of cardiac MRI images and the subsequent classification of heart diseases. It utilizes advanced Deep Learning and Machine Learning techniques, specifically trained and evaluated on the ACDC (Automated Cardiac Diagnosis Challenge) dataset.

## Project Overview

The project is divided into two main tasks:
1. **Segmentation**: Identifying and delineating three cardiac structures (Right Ventricle, Myocardium, and Left Ventricle) in 3D MRI scans. For this, the **MedSAM2** model is fine-tuned.
2. **Classification**: Predicting a patient's heart disease (Normal, MINF, DCM, HCM, RV) using Machine Learning models (Random Forest, XGBoost, CatBoost). These models use clinical features (such as volumes and ejection fraction) extracted from the segmentation masks.

## File Structure and Modules

Below is an overview of the main scripts in this repository and their functions:

* **`segmentation.py`**
  Contains the full training loop for the MedSAM2 segmentation model. The script uses PyTorch and implements data augmentation (rotations, flipping), mixed-precision training (bfloat16), and cross-validation (GroupKFold). It also includes a custom `CombinedLoss` and calculates 3D Dice and HD95 scores for medical evaluation.

* **`Classification.ipynb`**
  A comprehensive Jupyter Notebook where predicted clinical data (such as ES/ED volumes and Ejection Fraction) is combined with patient metadata (weight, height) for classification. Multiple tree-based models (RandomForest, XGBoost, CatBoost) are trained. The notebook also features SHAP feature importance visualizations to explain model decisions.

* **`testingscore.py`**
  This script evaluates the generated predictions by comparing them with the ground truth. In addition to geometric metrics (Dice and HD95), it calculates key clinical values such as Volume and Ejection Fraction (EF). It then computes the Pearson correlation between predictions and reality.

* **`imagemaskproduction.py`**
  A utility script for visualization. It takes the original NIfTI MRI images and overlays the predicted segmentation masks (RV, MYO, LV) in color. Very useful for quick visual quality control of the predictions.

* **`metricsoverwriter.py`**
  Generates the standard `metrics.py` script (originally by Clément Zotti). This provides handy NIfTI I/O helper functions (`load_nii`, `save_nii`) and built-in ACDC validation functions.

* **`creatingtestfiles`**
  A script related to generating and formatting test files for the pipeline process.

## Installation & Requirements

Ensure Python (3.x) is installed along with the necessary packages. It is recommended to use a virtual environment.

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn
pip install nibabel medpy matplotlib tqdm
pip install xgboost catboost shap
pip install hydra-core
