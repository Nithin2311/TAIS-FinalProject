# TAIS-MidtermProject

# Resume Classification System - Midterm Project
## CAI 6605: Trustworthy AI Systems | Fall 2025 | Group 15

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-84.45%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

An AI-powered resume classification system that automatically categorizes resumes into **24 job categories** with **84.45% test accuracy**, exceeding the 80% target requirement. Built with RoBERTa-base and featuring an automated data pipeline with professional Gradio interface.

### 🎯 Key Achievements
- ✅ **84.45% Test Accuracy** (Target: >80%)
- ✅ **Automated Dataset Pipeline** from Google Drive
- ✅ **Professional Web Interface** with real-time predictions
- ✅ **Modular Architecture** ready for bias detection (final project)
- ✅ **Comprehensive Evaluation** with per-category performance metrics

## 👥 Team Members & Contributions

| Team Member | Contribution | Percentage |
|-------------|--------------|------------|
| **Nithin Palyam** | Model architecture, training pipeline optimization, performance tuning, system integration | 50% |
| **Lorenzo LaPlace** | Data preprocessing pipeline, Gradio interface development, documentation, automated dataset handling | 50% |

*Each member contributed equally to project planning, testing, and evaluation.*

## 🚀 Quick Start Guide

### Prerequisites
* - Python 3.8+
* - Google Colab (recommended) or local environment with 4GB+ GPU RAM

### Installation & Setup

# Option 1: Google Colab (Recommended for Best Performance)
bash
# Clone repository
* !git clone https://github.com/Nithin2311/TAIS-MidtermProject.git
* %cd TAIS-MidtermProject

# Install dependencies
!pip install -r requirements.txt

# Run training (automatically downloads dataset)
!python train.py

# Launch web interface
!python gradio_app.py

# Option 2: Local Installation

# Clone repository
* git clone https://github.com/Nithin2311/TAIS-MidtermProject.git
* cd TAIS-MidtermProject

# Create virtual environment (recommended)
* python -m venv venv
* source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Launch web interface
python gradio_app.py

# Source Declaration
* Dataset:	resume-dataSet from kaggle Academic Use	Used as-is for academic purposes
* Base Model:	RoBERTa-base from HuggingFace	MIT License	Fine-tuned on resume data
* Libraries: 	PyTorch, Transformers, Gradio, Scikit-learn	Various Open Source	Used according to license terms

# Original Code
* Data preprocessing pipeline - Original implementation,
* Automated dataset download system - Original implementation,
* Model training framework - Original implementation,
* Gradio interface design - Original implementation,
* Evaluation metrics system - Original implementation,
* Project architecture - Original design,
* All core machine learning components and system architecture were implemented from scratch by the team members.

# Troubleshooting
* Common Issues & Solutions
* -CUDA Out of Memory:	Reduce batch size to 8 in config.py
* -Dataset Download Failed:	Check internet connection; URL: https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk
* -Import Errors:	Run pip install -r requirements.txt
* -Gradio Not Loading:	Check firewall settings or use share=True in launch()

# Performance Tips
* -Use Google Colab Pro for faster GPU training
* -Enable FP16 training in config.py
* -Reduce MAX_LENGTH to 256 for faster inference
* -Use batch processing for multiple resumes

# References
* Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset from kaggle 
* RoBERTa Paper: Liu et al. (2019) "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
* HuggingFace Transformers: Wolf et al. (2020) "Transformers: State-of-the-Art Natural Language Processing"
* Course Materials: CAI 6605 Trustworthy AI Systems, USF Fall 2025

# License: 
* This project is for academic purposes as part of CAI 6605: Trustworthy AI Systems at the University of South Florida. All rights reserved by the course instructors and team members.

# Course Information
* Course: CAI 6605 - Trustworthy AI Systems
* University: University of South Florida
* Semester: Fall 2025
* Instructor: Guangjing Wang
* Team: Group 15 (Nithin Palyam, Lorenzo LaPlace)
* Submission Date: October 2025


