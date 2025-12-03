# Enhanced Resume Classification System with Comprehensive Bias Mitigation

**CAI 6605: Trustworthy AI Systems – Final Project**  
*University of South Florida, Fall 2025*  
**Group 15**: Nithin Palyam, Lorenzo LaPlace

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![Gradio](https://img.shields.io/badge/Gradio-3.35+-green.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project implements a trustworthy AI system for resume classification that not only achieves high accuracy but also actively detects and mitigates demographic biases. The system features:

- **Dual-model architecture**: Baseline vs. Debiased models for comparison
- **Comprehensive bias mitigation**: Multiple debiasing techniques (preprocessing, in-processing, post-processing)
- **Explainability**: Integrated LIME explanations for model predictions
- **Fairness evaluation**: 10+ fairness metrics including demographic parity, equal opportunity, and intersectional fairness
- **Interactive interface**: Gradio web app for real-time testing and comparison

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **High Accuracy** | 88.19% baseline accuracy on 24 job categories |
| **Bias Reduction** | 100% elimination of name-based gender bias |
| **Multi-Attribute Fairness** | Gender, race, age, educational privilege, disability |
| **Explainability** | LIME explanations for model decisions |
| **Interactive Demo** | Gradio interface with real-time bias analysis |
| **Open Source** | Fully auditable codebase for transparency |

## 📊 Performance Summary

| Metric | Baseline | Debiased | Improvement |
|--------|----------|----------|-------------|
| Accuracy | 88.19% | 87.73% | -0.46% (minimal trade-off) |
| Gender Bias | 0.067 | 0.000 | **100% reduction** |
| Racial Bias | 0.400 | 0.133 | 66.75% reduction |
| Intersectional Fairness | 0.556 | 0.750 | +34.89% improvement |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM
- GPU recommended (4GB+ VRAM) but not required
- 5GB free disk space

### Installation

1. **Clone the repository**:
bash
git clone https://github.com/Nithin2311/TAIS-FinalProject.git
cd TAIS-FinalProject

2. Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

bash
pip install -r requirements.txt

Download NLTK data (for text processing):

bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
🏃‍♂️ Running the System
Step 1: Train the Baseline Model
bash
python train_baseline.py
Downloads the dataset automatically

Trains RoBERTa-base on resume data

Saves model to models/resume_classifier_baseline/

Generates results in results/baseline_results.json

Expected Output: ~88% accuracy, training time ~2 hours on Tesla T4 GPU

Step 2: Train the Debiased Model
bash
python train_debiased.py
Applies bias mitigation techniques

Uses adversarial training and counterfactual augmentation

Saves model to models/resume_classifier_debiased/

Generates results in results/debiased_results.json

Expected Output: ~87.7% accuracy with reduced bias

Step 3: Run Bias Analysis
bash
python bias_analysis.py
Compares baseline vs. debiased models

Calculates 10+ fairness metrics

Generates comprehensive bias report

Saves comparison to results/model_comparison.json

Expected Output: Bias reduction statistics and fairness analysis

Step 4: Launch the Web Interface
bash
python gradio_app.py
Starts Gradio web server

Access at http://localhost:7860

Compare both models in real-time

View LIME explanations and bias scores

📁 Project Structure
text
TAIS-FinalProject/
├── models/                          # Trained models
│   ├── resume_classifier_baseline/  # Baseline model
│   └── resume_classifier_debiased/  # Debiased model
├── data/                            # Dataset
│   ├── raw/                         # Original data
│   └── processed/                   # Processed data
├── results/                         # Evaluation results
├── config.py                        # Configuration settings
├── data_processor.py                # Data preprocessing
├── model_trainer.py                 # Model training utilities
├── train_baseline.py                # Baseline training script
├── train_debiased.py                # Debiased training script
├── bias_analyzer.py                 # Bias detection and analysis
├── bias_analysis.py                 # Model comparison script
├── fairness_metrics.py              # Fairness metrics calculation
├── demographic_synthesizer.py       # Demographic signal generation
├── gradio_app.py                    # Web interface
└── requirements.txt                 # Dependencies
🔧 Configuration
Key parameters in config.py:

Parameter	Default Value	Description
MODEL_NAME	'roberta-base'	Base transformer model
BATCH_SIZE	16	Training batch size
LEARNING_RATE	3e-5	Learning rate
ADVERSARIAL_LAMBDA	0.03	Debiasing strength
NUM_EPOCHS	15	Training epochs
MAX_LENGTH	512	Maximum sequence length
📊 Dataset
Source: Updated Resume Dataset (Kaggle)

Statistics:

2,484 resumes

24 job categories

Split: 70% train, 15% validation, 15% test

Categories: ACCOUNTANT, ADVOCATE, AGRICULTURE, ..., TEACHER

Preprocessing:

Text cleaning and normalization

Demographic signal extraction

Counterfactual augmentation for underperforming categories

Fairness-aware sample weighting

🤖 Model Architecture
Baseline Model
Base: RoBERTa-base (125M parameters)

Loss: Weighted Cross-Entropy + Focal Loss

Regularization: Dropout (0.25), Gradient clipping

Optimizer: AdamW with linear warmup

Debiased Model
Additional Components:

Correlation Penalty Loss (λ=0.025)

Demographic signal extraction

Counterfactual augmentation

Fairness-aware sample reweighting

Training: Adversarial debiasing with early stopping

⚖️ Fairness Metrics Implemented
The system evaluates 10+ fairness metrics:

Demographic Parity Difference

Equal Opportunity Difference

Disparate Impact Ratio

Accuracy Equality Difference

Equalized Odds Difference

Treatment Equality Ratio

Intersectional Fairness

Name Substitution Bias

Counterfactual Fairness

Statistical Parity

🌐 Web Interface Features
The Gradio interface (gradio_app.py) provides:

Dual-Model Comparison: Switch between baseline and debiased models

Real-Time Classification: Paste any resume text for instant classification

Bias Analysis: View inferred demographics and bias scores

LIME Explanations: Visual feature importance for predictions

Example Resumes: Pre-loaded examples for quick testing

Performance Metrics: Side-by-side model comparison

📈 Results Interpretation
Reading Output Files
results/baseline_results.json: Baseline model performance

results/debiased_results.json: Debiased model performance

results/model_comparison.json: Detailed bias comparison

results/training_results.json: Per-category accuracy analysis

Key Metrics to Monitor
Metric	Ideal Value	Our Results
Accuracy	>80%	87.73%
Gender Bias	0.000	✅ 0.000
Racial Bias	<0.100	0.133
Intersectional Fairness	>0.700	✅ 0.750
Accuracy-Fairness Trade-off	<2%	✅ 0.46%
🛠️ Troubleshooting
Common Issues
"CUDA out of memory"

bash
# Reduce batch size in config.py
BATCH_SIZE = 8  # Change from 16 to 8
"Model not found" errors

bash
# Ensure models are trained first
python train_baseline.py
python train_debiased.py
Slow training

bash
# Enable GPU if available
# Reduce MAX_LENGTH in config.py
MAX_LENGTH = 256  # Change from 512
Missing dependencies

bash
# Update pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt
Memory Requirements
Component	Minimum	Recommended
RAM	8GB	16GB
GPU VRAM	4GB	8GB+
Disk Space	5GB	10GB
📚 Citation
If you use this project in your research, please cite:

bibtex
@software{TAIS_Resume_Classifier_2025,
  author = {Palyam, Nithin and LaPlace, Lorenzo},
  title = {Enhanced Resume Classification System with Comprehensive Bias Mitigation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Nithin2311/TAIS-FinalProject}}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👥 Team Contributions
Nithin Palyam:

Model architecture and training pipeline

Bias mitigation implementation

Fairness metrics calculation

Documentation and testing

Lorenzo LaPlace:

Data preprocessing pipeline

Gradio interface development

LIME/SHAP explainability features

System evaluation and validation

🔗 Useful Links
Project Repository

Hugging Face Models

Gradio Documentation

AI Fairness 360 Toolkit

Trustworthy AI Guidelines

🙏 Acknowledgments
Dataset: Kaggle Resume Dataset

Base Model: RoBERTa by Facebook AI

Fairness Framework: Inspired by IBM AI Fairness 360

Course: CAI 6605 Trustworthy AI Systems, University of South Florida
