# HP at BAREC Shared Task 2026: Parameter-Efficient Ordinal Regression and Threshold Optimization for Arabic Readability Assessment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the system submitted by **Team HP** for the **BAREC Shared Task 2026** on Fine-grained Arabic Readability Assessment (Strict Track). 

Our approach reframes the 19-level readability task as an **Ordinal Regression** problem to directly optimize the Quadratic Weighted Kappa (QWK) metric. By combining Parameter-Efficient Fine-Tuning (LoRA), the CORAL loss framework, and an advanced Post-Processing ensemble (Nelder-Mead optimization + Distribution Alignment), our system achieved a highly competitive QWK score of **0.8370** on the official blind test.

## Key Features
* **Parameter-Efficient Fine-Tuning:** Utilizing Low-Rank Adaptation (LoRA) on AraBERTv02 and ARBERT to prevent overfitting on imbalanced data.
* **Ordinal Regression (CORAL Framework):** Learning 18 hierarchical binary boundaries instead of standard nominal classification, effectively penalizing distant misclassifications.
* **Nelder-Mead Threshold Optimization:** Empirically searching for the 18 optimal decision boundaries on Out-Of-Fold (OOF) logits.
* **Distribution Alignment & Soft Blending:** A 50/50 soft blending strategy to counteract conservative prediction behaviors and perfectly mirror the real-world label distribution.

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
│   ├── 03_Main_AraBERT_CORAL_5Fold.ipynb
│   └── 04_Main_ARBERT_Baseline.ipynb
│
├── requirements.txt        
└── README.md