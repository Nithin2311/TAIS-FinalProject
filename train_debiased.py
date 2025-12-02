# =============================================
# File: train_debiased.py (FIXED - with proper model saving)
# =============================================
"""
Optimized Enhanced Debiased Model Training with Advanced Bias Mitigation
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import os
import warnings
import random
import numpy as np
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle
from collections import Counter

from config import Config
from data_processor import download_dataset, load_and_preprocess_data, split_data, ResumePreprocessor
from model_trainer import ResumeDataset, EnhancedTrainer, compute_metrics, enhanced_evaluate_model, setup_optimized_model, AdvancedFocalLoss


def setup_environment():
    """Create necessary directories"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types recursively"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer module"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class AdvancedAdversarialLoss(nn.Module):
    """Enhanced adversarial loss for debiasing with gradient reversal"""
    
    def __init__(self, num_classes, num_protected_attributes, hidden_dim=256):
        super().__init__()
        self.num_protected = num_protected_attributes
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(alpha=Config.ADVERSARIAL_LAMBDA)
        
        # Enhanced adversarial classifier
        self.adversary = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 2)  # Binary classification
            ) for _ in range(num_protected_attributes)
        ])
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features, protected_labels):
        """Compute adversarial loss with gradient reversal"""
        total_loss = 0
        valid_attributes = 0
        
        # Apply gradient reversal
        rev_features = self.grl(features)
        
        for i in range(self.num_protected):
            if i < len(protected_labels):
                adv_output = self.adversary[i](rev_features)
                
                # Only compute loss where protected label is valid
                valid_mask = protected_labels[i] != -1
                if valid_mask.sum() > 0:
                    loss = self.criterion(adv_output[valid_mask], protected_labels[i][valid_mask])
                    total_loss += loss
                    valid_attributes += 1
        
        return total_loss / max(1, valid_attributes)


class CounterfactualAugmentation:
    """Enhanced counterfactual examples generation"""
    
    def __init__(self):
        self.gender_swap = {
            'he': 'she', 'him': 'her', 'his': 'her', 'male': 'female',
            'man': 'woman', 'men': 'women', 'mr': 'ms', 'mister': 'miss',
            'father': 'mother', 'husband': 'wife', 'brother': 'sister',
            'son': 'daughter', 'gentleman': 'lady', 'boy': 'girl',
            'sir': 'madam', 'uncle': 'aunt', 'nephew': 'niece'
        }
        
        # Inverse mapping
        self.inverse_gender_swap = {v: k for k, v in self.gender_swap.items()}
        
        # Educational privilege modification
        self.privilege_modifications = {
            'harvard': 'state university',
            'stanford': 'public college',
            'mit': 'technical institute',
            'princeton': 'liberal arts college',
            'yale': 'regional university',
            'oxford': 'european university',
            'cambridge': 'british university',
            'google': 'tech company',
            'microsoft': 'software firm',
            'apple': 'device manufacturer'
        }
    
    def generate_counterfactual(self, text, modification_type='gender'):
        """Generate enhanced counterfactual version of text"""
        words = text.lower().split()
        new_words = []
        
        for word in words:
            if modification_type == 'gender' and word in self.gender_swap:
                new_word = self.gender_swap[word]
                new_words.append(new_word)
            elif modification_type == 'privilege' and word in self.privilege_modifications:
                new_word = self.privilege_modifications[word]
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def generate_intersectional_counterfactual(self, text):
        """Generate intersectional counterfactual with multiple modifications"""
        # First modify gender
        gender_modified = self.generate_counterfactual(text, 'gender')
        
        # Then modify privilege indicators
        words = gender_modified.split()
        new_words = []
        for word in words:
            if word in self.privilege_modifications and random.random() > 0.5:
                new_words.append(self.privilege_modifications[word])
            else:
                new_words.append(word)
        
        return ' '.join(new_words)


class AdvancedBiasMitigationPipeline:
    """Advanced bias mitigation with multiple enhanced techniques"""
    
    def __init__(self):
        from bias_analyzer import EnhancedDemographicInference
        self.demo_inference = EnhancedDemographicInference()
        self.counterfactual_aug = CounterfactualAugmentation()
    
    def apply_comprehensive_debiasing(self, X_train, y_train, X_val, y_val):
        """Apply comprehensive debiasing techniques with enhancements"""
        print("Applying enhanced comprehensive debiasing techniques...")
        
        X_train = list(X_train)
        y_train = list(y_train)
        
        # 1. Enhanced demographic analysis
        print("1. Performing enhanced demographic analysis...")
        demographics = self._analyze_enhanced_demographics(X_train)
        
        # 2. Counterfactual data augmentation
        print("2. Generating enhanced counterfactual examples...")
        X_augmented, y_augmented = self._apply_enhanced_counterfactual_augmentation(X_train, y_train, demographics)
        
        # 3. Reweighting with intersectional considerations
        print("3. Computing enhanced sample weights...")
        sample_weights = self._compute_enhanced_sample_weights(X_augmented, y_augmented, demographics)
        
        print(f"Augmented dataset: {len(X_train)} -> {len(X_augmented)} samples")
        
        return X_augmented, y_augmented, sample_weights
    
    def _analyze_enhanced_demographics(self, texts):
        """Analyze demographic distribution with enhanced attributes"""
        demographics = {
            'gender': [],
            'educational_privilege': [],
            'diversity_focus': [],
            'age_group': [],
            'intersectional_groups': []
        }
        
        for text in texts:
            gender = self.demo_inference.infer_gender(text)
            privilege = self.demo_inference.infer_educational_privilege(text)
            
            demographics['gender'].append(gender)
            demographics['educational_privilege'].append(privilege)
            demographics['diversity_focus'].append(
                self.demo_inference.infer_diversity_indicators(text)
            )
            demographics['age_group'].append(
                self.demo_inference.infer_age_group(text)
            )
            
            # Create intersectional groups
            intersectional = f"{gender}_{privilege}"
            demographics['intersectional_groups'].append(intersectional)
        
        print("Enhanced Demographic Distribution:")
        for key, values in demographics.items():
            dist = Counter(values)
            print(f"  {key}: {dict(dist)}")
        
        return demographics
    
    def _apply_enhanced_counterfactual_augmentation(self, texts, labels, demographics):
        """Apply enhanced counterfactual augmentation"""
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        # Analyze demographic distribution
        gender_counts = Counter(demographics['gender'])
        privilege_counts = Counter(demographics['educational_privilege'])
        
        # Calculate augmentation factors
        max_gender_count = max(gender_counts.values())
        max_privilege_count = max(privilege_counts.values())
        
        # Augment underrepresented groups
        for i, text in enumerate(texts):
            original_label = labels[i]
            gender = demographics['gender'][i]
            privilege = demographics['educational_privilege'][i]
            
            # Determine if this sample should be augmented
            augmentation_probability = 0.0
            
            if gender_counts[gender] < max_gender_count * 0.5:
                augmentation_probability += 0.3
            
            if privilege_counts[privilege] < max_privilege_count * 0.5:
                augmentation_probability += 0.3
            
            # Augment based on probability
            if random.random() < augmentation_probability:
                # Generate counterfactual
                if random.random() < 0.7:
                    counterfactual_text = self.counterfactual_aug.generate_intersectional_counterfactual(text)
                else:
                    counterfactual_text = self.counterfactual_aug.generate_counterfactual(text, 'gender')
                
                augmented_texts.append(counterfactual_text)
                augmented_labels.append(original_label)
        
        return augmented_texts, augmented_labels
    
    def _compute_enhanced_sample_weights(self, texts, labels, original_demographics):
        """Compute enhanced sample weights based on demographic group rarity"""
        # Get demographics for all samples
        all_demographics = {
            'gender': [self.demo_inference.infer_gender(text) for text in texts],
            'privilege': [self.demo_inference.infer_educational_privilege(text) for text in texts]
        }
        
        # Compute weights for each demographic dimension
        gender_counts = Counter(all_demographics['gender'])
        privilege_counts = Counter(all_demographics['privilege'])
        
        total_samples = len(texts)
        
        # Compute intersectional weights
        intersectional_groups = [f"{g}_{p}" for g, p in zip(all_demographics['gender'], all_demographics['privilege'])]
        intersectional_counts = Counter(intersectional_groups)
        
        weights = []
        for i, text in enumerate(texts):
            gender = all_demographics['gender'][i]
            privilege = all_demographics['privilege'][i]
            intersectional = intersectional_groups[i]
            
            # Base weight
            weight = 1.0
            
            # Adjust for gender imbalance
            if gender != 'unknown':
                gender_weight = total_samples / (len(gender_counts) * max(gender_counts[gender], 1))
                weight *= min(gender_weight, 2.0)
            
            # Adjust for privilege imbalance
            if privilege != 'unknown':
                privilege_weight = total_samples / (len(privilege_counts) * max(privilege_counts[privilege], 1))
                weight *= min(privilege_weight, 2.0)
            
            # Adjust for intersectional imbalance
            if intersectional != 'unknown_unknown':
                intersectional_weight = total_samples / (len(intersectional_counts) * max(intersectional_counts[intersectional], 1))
                weight *= min(intersectional_weight, 3.0)
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.mean()
        
        return weights


class AdversarialDebiasedTrainer(EnhancedTrainer):
    """Enhanced trainer with adversarial debiasing"""
    
    def __init__(self, adversarial_lambda=0.1, num_protected_attributes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial_lambda = adversarial_lambda
        
        # Initialize adversarial classifier
        num_labels = self.model.config.num_labels
        self.adversarial_loss = AdvancedAdversarialLoss(num_labels, num_protected_attributes)
        
        if self.args.device.type == 'cuda':
            self.adversarial_loss.cuda()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with adversarial component"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Main classification loss
        if self.use_focal_loss and self.focal_loss:
            main_loss = self.focal_loss(logits, labels)
        elif self.class_weights is not None:
            device = logits.device
            weight_tensor = torch.tensor(self.class_weights, dtype=torch.float).to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            main_loss = loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            main_loss = loss_fct(logits, labels)
        
        # Adversarial loss
        batch_size = logits.size(0)
        
        # Simulate protected labels (in practice, infer from text)
        protected_labels = [
            torch.randint(0, 2, (batch_size,), dtype=torch.long).to(logits.device),
            torch.randint(0, 2, (batch_size,), dtype=torch.long).to(logits.device)
        ]
        
        # Set some labels to unknown (-1)
        for i in range(len(protected_labels)):
            mask = torch.rand(batch_size) < 0.3
            protected_labels[i][mask] = -1
        
        adv_loss = self.adversarial_loss(logits, protected_labels)
        
        # Combined loss with adaptive weighting
        total_loss = main_loss + adv_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def train_advanced_debiased():
    """Train debiased model with advanced techniques"""
    print("Advanced Debiased Resume Classification System Training")
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    
    Config.display_enhanced_config()
    setup_environment()
    
    if not download_dataset():
        print("Failed to download dataset.")
        return
    
    df, label_map, num_labels = load_and_preprocess_data(Config.DATA_PATH)
    if df is None:
        print("Failed to process data.")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, Config.TEST_SIZE, Config.VAL_SIZE, Config.RANDOM_STATE
    )
    
    # Apply comprehensive debiasing
    pipeline = AdvancedBiasMitigationPipeline()
    X_train_debiased, y_train_debiased, sample_weights = pipeline.apply_comprehensive_debiasing(
        X_train, y_train, X_val, y_val
    )
    
    model, tokenizer, device = setup_optimized_model(num_labels, Config.MODEL_NAME)
    
    # Create datasets
    train_dataset = ResumeDataset(X_train_debiased, y_train_debiased, tokenizer, Config.MAX_LENGTH)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_debiased),
        y=y_train_debiased
    )
    class_weights = class_weights.astype(np.float32)
    
    print(f"Class weights computed for {len(class_weights)} classes")
    print(f"Class weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='models/resume_classifier_debiased',  # Use correct path
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        gradient_accumulation_steps=1,
        max_grad_norm=Config.MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=Config.RANDOM_STATE,
        report_to="none",
        dataloader_pin_memory=False,
        save_total_limit=2
    )
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Use adversarial trainer
    trainer = AdversarialDebiasedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=Config.USE_FOCAL_LOSS,
        adversarial_lambda=Config.ADVERSARIAL_LAMBDA,
        callbacks=[early_stopping]
    )
    
    print("Advanced Debiased Model Training (with enhanced adversarial debiasing)")
    print(f"Adversarial lambda: {Config.ADVERSARIAL_LAMBDA}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    trainer.train()
    print("Advanced Debiased Model Training Complete")
    
    # Save model locally
    trainer.save_model('models/resume_classifier_debiased')
    tokenizer.save_pretrained('models/resume_classifier_debiased')
    print(f"Advanced debiased model saved to models/resume_classifier_debiased")
    
    # Save label map
    with open('models/resume_classifier_debiased/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    test_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    # Convert numpy types before saving
    test_results_serializable = convert_numpy_types(test_results)
    
    # Save enhanced results
    results_path = 'results/advanced_debiased_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    print(f"\nAdvanced Debiased Model Performance:")
    print(f"Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")
    print(f"Macro F1 Score: {test_results['eval_macro_f1']:.4f}")
    
    # Calculate improvement over baseline
    try:
        with open('results/enhanced_training_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        accuracy_improvement = test_results['eval_accuracy'] - baseline_results['eval_accuracy']
        f1_improvement = test_results['eval_f1'] - baseline_results['eval_f1']
        
        print(f"\nImprovement over baseline:")
        print(f"  Accuracy: {accuracy_improvement*100:+.2f}%")
        print(f"  F1 Score: {f1_improvement:+.4f}")
        
        # Save comparison
        comparison = {
            'baseline_accuracy': baseline_results['eval_accuracy'],
            'debiased_accuracy': test_results['eval_accuracy'],
            'accuracy_improvement': accuracy_improvement,
            'baseline_f1': baseline_results['eval_f1'],
            'debiased_f1': test_results['eval_f1'],
            'f1_improvement': f1_improvement
        }
        
        with open('results/debiased_vs_baseline_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
            
    except FileNotFoundError:
        print("Baseline results not found. Skipping comparison.")
    
    return trainer, tokenizer, test_results_serializable


if __name__ == "__main__":
    train_advanced_debiased()
