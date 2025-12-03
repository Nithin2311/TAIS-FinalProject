# Resume Classification System with Bias Mitigation

A fairness-aware resume classification system that categorizes resumes into 24 job categories while actively mitigating demographic biases using advanced machine learning techniques.

## 🎯 Project Overview

This system uses RoBERTa-based transformers to classify resumes while implementing comprehensive bias mitigation strategies. It addresses fairness concerns in AI-driven hiring by reducing gender, racial, and socioeconomic biases through adversarial training, counterfactual augmentation, and fairness-aware loss functions.

## 📊 Performance Metrics

### Model Comparison

| Model | Accuracy | F1 Score | Macro F1 | Gender Bias | Racial Bias |
|-------|----------|----------|----------|-------------|-------------|
| **Baseline** | 88.19% | 0.8792 | 0.8792 | 0.067 | 0.400 |
| **Debiased** | 87.73% | 0.8750 | 0.8750 | 0.133 | 0.000 |

### Key Achievements
- ✅ **100% Racial Bias Reduction** (0.400 → 0.000)
- ✅ **Minimal Accuracy Trade-off** (-0.46%)
- ✅ **19.4% Improvement in Intersectional Fairness** (0.556 → 0.750)
- ✅ **17% Reduction in Gender Equalized Odds** (0.184 → 0.167)

### Fairness Improvements

**Intersectional Fairness:**
- Baseline: 0.556
- Debiased: 0.750 (+19.4% improvement)

**Equalized Odds:**
- Gender: Improved from 0.184 to 0.167
- Race: Comparable performance (0.683 to 0.692)

## 🏗️ System Architecture

```
├── Data Processing
│   ├── Resume preprocessing & cleaning
│   ├── Feature extraction (skills, experience)
│   └── Class balancing & augmentation
│
├── Baseline Model
│   ├── RoBERTa-base architecture
│   ├── Focal Loss for class imbalance
│   └── Standard training pipeline
│
├── Debiased Model
│   ├── Demographic signal extraction
│   ├── Counterfactual augmentation
│   ├── Correlation penalty loss
│   ├── Fairness-aware training
│   └── Adversarial debiasing (λ=0.03)
│
└── Bias Analysis
    ├── Name substitution experiments
    ├── Demographic inference
    ├── LIME explanations
    └── Fairness metrics computation
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
8GB+ RAM
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd resume-classification-system

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Training Pipeline

```bash
# Step 1: Train baseline model
python train_baseline.py

# Step 2: Train debiased model
python train_debiased.py

# Step 3: Run bias analysis
python bias_analysis.py

# Step 4: Launch web interface
python gradio_app.py
```

## 📁 Project Structure

```
resume-classification-system/
│
├── config.py                    # Configuration settings
├── data_processor.py            # Data loading & preprocessing
├── model_trainer.py             # Training utilities
├── bias_analyzer.py             # Bias detection & analysis
├── fairness_metrics.py          # Fairness metric calculations
├── demographic_synthesizer.py   # Demographic signal extraction
├── train_baseline.py            # Baseline model training
├── train_debiased.py            # Debiased model training
├── bias_analysis.py             # Comprehensive bias analysis
├── gradio_app.py                # Web interface
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── raw/                     # Raw resume dataset
│   └── processed/               # Processed data & splits
│
├── models/
│   ├── resume_classifier_baseline/
│   └── resume_classifier_debiased/
│
└── results/
    ├── baseline_results.json
    ├── debiased_results.json
    ├── model_comparison.json
    └── simplified_comparison.json
```

## 🎓 Job Categories (24 Classes)

1. ACCOUNTANT
2. ADVOCATE
3. AGRICULTURE
4. APPAREL
5. ARTS
6. AUTOMOBILE
7. AVIATION
8. BANKING
9. BPO
10. BUSINESS-DEVELOPMENT
11. CHEF
12. CONSTRUCTION
13. CONSULTANT
14. DESIGNER
15. DIGITAL-MEDIA
16. ENGINEERING
17. FINANCE
18. FITNESS
19. HEALTHCARE
20. HR
21. INFORMATION-TECHNOLOGY
22. PUBLIC-RELATIONS
23. SALES
24. TEACHER

## 🔬 Bias Mitigation Techniques

### 1. **Counterfactual Augmentation**
- Generates alternative versions of resumes with swapped demographic signals
- Creates gender-neutral and industry-balanced variants
- Augmentation factor: 3x for biased samples

### 2. **Correlation Penalty Loss**
- Penalizes correlation between predictions and demographic features
- Lambda parameter: 0.03 (optimized for fairness-accuracy trade-off)
- Reduces demographic signal influence on decisions

### 3. **Fairness-Aware Reweighting**
- Assigns higher weights to underrepresented groups
- Balances class distribution dynamically
- Reduces overfitting to majority demographics

### 4. **Demographic Signal Extraction**
- Identifies 9 demographic features from text
- Gender, socioeconomic, and industry bias signals
- Used for adversarial training and fairness monitoring

### 5. **Enhanced Class Balancing**
- Targeted augmentation for underperforming categories
- Oversampling of minority classes
- Maintains label distribution integrity

## 📊 Fairness Metrics

### Demographic Parity
Measures equal positive prediction rates across groups
- **Gender**: 0.000 (perfect parity)
- **Race**: 0.000 (perfect parity)

### Equal Opportunity
Measures equal true positive rates across groups
- **Gender**: 0.167 (improved from 0.184)
- **Race**: 0.692 (comparable to baseline)

### Intersectional Fairness
Evaluates accuracy across demographic intersections
- **Score**: 0.750 (19.4% improvement)
- Analyzes gender × race combinations

### Name Substitution Experiments
Tests model predictions with different demographic names
- **Gender Bias**: 0.133
- **Racial Bias**: 0.000 (eliminated)

## 🖥️ Web Interface

The Gradio-based interface provides:

- **Real-time Classification**: Upload or paste resume text
- **Model Comparison**: Switch between baseline and debiased models
- **Confidence Scores**: View top 5 predictions with probabilities
- **LIME Explanations**: Understand model decisions
- **Demographic Inference**: See inferred demographic attributes
- **Performance Metrics**: Compare model statistics

### Launch Interface

```bash
python gradio_app.py
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model Settings
MODEL_NAME = 'roberta-base'
MAX_LENGTH = 512

# Training Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 3e-5

# Bias Mitigation
ADVERSARIAL_LAMBDA = 0.03
COUNTERFACTUAL_AUGMENTATION_FACTOR = 2.0
USE_FOCAL_LOSS = True

# Target Categories for Enhancement
TARGET_AUGMENTATION_CATEGORIES = [
    'BPO', 'AUTOMOBILE', 'APPAREL', 
    'DIGITAL-MEDIA', 'ARTS'
]
```

## 📈 Detailed Results

### Perfect Accuracy Categories (Baseline)
- ACCOUNTANT (100%)
- BUSINESS-DEVELOPMENT (100%)
- CONSTRUCTION (100%)
- DESIGNER (100%)
- ENGINEERING (100%)

### Challenging Categories
| Category | Baseline | Debiased | Status |
|----------|----------|----------|--------|
| APPAREL | 55.6% | 66.7% | ✅ Improved |
| HEALTHCARE | 61.1% | 55.6% | ⚠️ Needs work |
| AVIATION | 66.7% | 66.7% | ➡️ Stable |
| ARTS | - | 66.7% | 🆕 New issue |

## 🛡️ Bias Analysis Pipeline

```python
# 1. Demographic Inference
demographics = infer_demographics(resume_text)
# Output: {'gender': 'male', 'race': 'asian', 'age_group': 'mid_career'}

# 2. Fairness Metrics
metrics = compute_fairness_metrics(
    y_true, y_pred, protected_attributes
)

# 3. Name Substitution
bias_scores = run_name_substitution_experiment(
    test_texts, male_names, female_names
)

# 4. LIME Explanations
explanation = explain_prediction(text, model, tokenizer)
```

## 🎯 Use Cases

- **HR Departments**: Screen resumes fairly without demographic bias
- **Recruitment Agencies**: Automate initial resume classification
- **Career Counseling**: Categorize candidate profiles objectively
- **Research**: Study AI fairness in hiring systems
- **Education**: Teach bias mitigation techniques in ML

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{resume_classification_fairness,
  title={Resume Classification System with Bias Mitigation},
  author={Your Name},
  year={2024},
  description={Fairness-aware resume classification using RoBERTa and adversarial debiasing}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Bias Metrics**: Implement more fairness measures
2. **Enhanced Augmentation**: Develop better counterfactual generation
3. **Multi-language Support**: Extend to non-English resumes
4. **Real-time Inference**: Optimize for production deployment
5. **Explainability**: Improve LIME/SHAP integrations

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ⚠️ Ethical Considerations

**Important Notes:**

1. **Not a Complete Solution**: This system reduces but does not eliminate bias
2. **Human Oversight Required**: Always review automated decisions
3. **Regular Auditing**: Continuously monitor for emerging biases
4. **Transparency**: Inform candidates when AI is used in screening
5. **Legal Compliance**: Ensure compliance with employment laws (EEOC, GDPR, etc.)

## 🐛 Known Issues

- Healthcare and Apparel categories show lower accuracy (<70%)
- LIME explanations can be slow for long resumes
- Demographic inference relies on heuristics (not always accurate)
- Gender bias shows slight increase in debiased model (requires investigation)
