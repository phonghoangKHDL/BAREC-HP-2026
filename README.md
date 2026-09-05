# HP at BAREC Shared Task 2026: Parameter-Efficient Ordinal Regression and Threshold Optimization for Arabic Readability Assessment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the system submitted by **Team HP** for the **BAREC Shared Task 2026** on Fine-grained Arabic Readability Assessment (Strict Track).

Our approach reframes the 19-level readability task as an **Ordinal Regression** problem to directly optimize the Quadratic Weighted Kappa (QWK) metric. By combining Parameter-Efficient Fine-Tuning (LoRA), the CORAL loss framework, and an 18-threshold Nelder-Mead optimizer calibrated exclusively on Out-Of-Fold logits, our system achieved an official blind test QWK of **0.8372**.

## Key Features
* **Parameter-Efficient Fine-Tuning:** Utilizing Low-Rank Adaptation (LoRA) on AraBERTv02 to prevent overfitting on imbalanced data.
* **Ordinal Regression (CORAL Framework):** Learning 18 hierarchical binary boundaries instead of standard nominal classification, effectively penalizing distant misclassifications.
* **Multi-Seed 5-Fold Cross-Validation:** Maximizing training data utilization and generating robust Out-Of-Fold logits for post-processing calibration.
* **Nelder-Mead Threshold Optimization:** Jointly optimizing 18 decision boundaries by directly maximizing QWK on blended OOF logits — all hyperparameters frozen before any blind test inference.

---

## Repository Structure

```text
BAREC-HP-2026/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 00_EDA_and_Distribution.ipynb
│   ├── 01_Data_Cleaning_and_Normalization.ipynb
│   ├── 02a_Ablation_Focal_Loss.ipynb
│   ├── 02b_Ablation_Weighted_CE.ipynb
│   ├── 02c_Ablation_MAE_Regression.ipynb
│   ├── 02d_Ablation_Pure_EMD.ipynb
│   ├── 02e_Ablation_Pure_CORAL.ipynb       
│   ├── 02f_Ablation_ARBERT_CORAL.ipynb     
│   ├── 03_Main_AraBERT_CORAL_5Fold.ipynb   ← Main pipeline: 5-fold + Nelder-Mead → blind test
│  
│
├── requirements.txt
└── README.md