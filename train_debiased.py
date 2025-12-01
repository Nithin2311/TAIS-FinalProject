"""
Enhanced Debiased Model Training with Advanced Bias Mitigation
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
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle
from collections import Counter

from config import Config
from data_processor import download_dataset, load_and_preprocess_data, split_data, ResumePreprocessor
from model_trainer import ResumeDataset, EnhancedTrainer, compute_metrics, enhanced_evaluate_model, setup_optimized_model, FocalLoss
from bias_analyzer import EnhancedDemographicInference


def setup_environment():
    """Create necessary directories"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)


class AdvancedAdversarialLoss(nn.Module):
    """Adversarial loss for debiasing - predicts protected attributes"""
    
    def __init__(self, num_classes, num_protected_attributes, hidden_dim=128):
        super().__init__()
        self.num_protected = num_protected_attributes
        
        # Adversarial classifier for each protected attribute
        self.adversary = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 2)  # Binary classification for each attribute
            ) for _ in range(num_protected_attributes)
        ])
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features, protected_labels):
        """Compute adversarial loss"""
        total_loss = 0
        batch_size = features.size(0)
        
        # Create a gradient reversal layer (multiply by -λ during backward)
        rev_features = features.detach() * 1.0  # Detach first
        
        for i in range(self.num_protected):
            if i < len(protected_labels):
                adv_output = self.adversary[i](rev_features)
                
                # Only compute loss where protected label is not unknown
                valid_mask = protected_labels[i] != -1
                if valid_mask.sum() > 0:
                    loss = self.criterion(adv_output[valid_mask], protected_labels[i][valid_mask])
                    total_loss += loss
        
        return total_loss / max(1, self.num_protected)


class CounterfactualAugmentation:
    """Generate counterfactual examples by modifying demographic indicators"""
    
    def __init__(self):
        self.gender_swap = {
            'he': 'she', 'him': 'her', 'his': 'her', 'male': 'female',
            'man': 'woman', 'men': 'women', 'mr': 'ms', 'mister': 'miss',
            'father': 'mother', 'husband': 'wife', 'brother': 'sister',
            'son': 'daughter', 'gentleman': 'lady', 'boy': 'girl'
        }
        
        # Inverse mapping
        self.gender_swap.update({v: k for k, v in self.gender_swap.items()})
    
    def generate_counterfactual(self, text, target_gender=None):
        """Generate counterfactual version of text"""
        if target_gender is None:
            # Randomly choose to flip gender
            target_gender = 'female' if random.random() > 0.5 else 'male'
        
        words = text.lower().split()
        new_words = []
        
        for word in words:
            if word in self.gender_swap:
                # Apply case preservation
                new_word = self.gender_swap[word]
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)


class AdvancedBiasMitigationPipeline:
    """Advanced bias mitigation with multiple techniques"""
    
    def __init__(self):
        self.demo_inference = EnhancedDemographicInference()
        self.counterfactual_aug = CounterfactualAugmentation()
    
    def apply_comprehensive_debiasing(self, X_train, y_train, X_val, y_val):
        """Apply comprehensive debiasing techniques"""
        print("Applying comprehensive debiasing techniques...")
        
        X_train = list(X_train)
        y_train = list(y_train)
        
        # 1. Demographic analysis
        print("1. Analyzing demographic distribution...")
        demographics = self._analyze_demographics(X_train)
        
        # 2. Counterfactual data augmentation
        print("2. Generating counterfactual examples...")
        X_augmented, y_augmented = self._apply_counterfactual_augmentation(X_train, y_train, demographics)
        
        # 3. Reweighting minority samples
        print("3. Computing sample weights...")
        sample_weights = self._compute_sample_weights(X_augmented, demographics)
        
        print(f"Augmented dataset: {len(X_train)} → {len(X_augmented)} samples")
        
        return X_augmented, y_augmented, sample_weights
    
    def _analyze_demographics(self, texts):
        """Analyze demographic distribution"""
        demographics = {
            'gender': [],
            'educational_privilege': [],
            'diversity_focus': [],
            'age_group': []
        }
        
        for text in texts:
            demographics['gender'].append(self.demo_inference.infer_gender(text))
            demographics['educational_privilege'].append(
                self.demo_inference.infer_educational_privilege(text)
            )
            demographics['diversity_focus'].append(
                self.demo_inference.infer_diversity_indicators(text)
            )
            demographics['age_group'].append(
                self.demo_inference.infer_age_group(text)
            )
        
        print("Demographic distribution:")
        for key, values in demographics.items():
            dist = Counter(values)
            print(f"  {key}: {dict(dist)}")
        
        return demographics
    
    def _apply_counterfactual_augmentation(self, texts, labels, demographics):
        """Apply counterfactual augmentation"""
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        # Analyze gender distribution
        gender_counts = Counter(demographics['gender'])
        target_count = max(gender_counts.values())
        
        # Augment minority gender groups
        for gender, count in gender_counts.items():
            if gender != 'unknown' and count < target_count:
                needed = target_count - count
                gender_indices = [i for i, g in enumerate(demographics['gender']) if g == gender]
                
                if gender_indices:
                    for _ in range(needed):
                        idx = random.choice(gender_indices)
                        original_text = texts[idx]
                        original_label = labels[idx]
                        
                        # Generate counterfactual
                        counterfactual_text = self.counterfactual_aug.generate_counterfactual(original_text)
                        augmented_texts.append(counterfactual_text)
                        augmented_labels.append(original_label)
        
        return augmented_texts, augmented_labels
    
    def _compute_sample_weights(self, texts, original_demographics):
        """Compute sample weights based on demographic group rarity"""
        # Get demographics for all samples
        all_demographics = {
            'gender': [self.demo_inference.infer_gender(text) for text in texts]
        }
        
        # Compute inverse frequency weights
        gender_counts = Counter(all_demographics['gender'])
        total_samples = len(texts)
        
        # Inverse frequency weighting
        weights = []
        for gender in all_demographics['gender']:
            if gender == 'unknown':
                weight = 1.0
            else:
                # More weight for minority groups
                weight = total_samples / (len(gender_counts) * gender_counts[gender])
                weight = min(weight, 3.0)  # Cap at 3x
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.mean()
        
        return weights


class AdversarialDebiasedTrainer(EnhancedTrainer):
    """Enhanced trainer with adversarial debiasing"""
    
    def __init__(self, adversarial_lambda=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial_lambda = adversarial_lambda
        
        # Initialize adversarial classifier
        num_labels = self.model.config.num_labels
        self.adversarial_loss = AdvancedAdversarialLoss(num_labels, num_protected_attributes=1)
        
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
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float).to(model.device)
            )
            main_loss = loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            main_loss = loss_fct(logits, labels)
        
        # Adversarial loss (predict protected attributes)
        # We need to infer protected attributes from input texts
        # For simplicity, we'll use a placeholder for now
        batch_size = logits.size(0)
        protected_labels = [torch.full((batch_size,), -1, dtype=torch.long).to(model.device)]
        
        adv_loss = self.adversarial_loss(logits, protected_labels)
        
        # Combined loss
        total_loss = main_loss - self.adversarial_lambda * adv_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def train_advanced_debiased():
    """Train debiased model with advanced techniques"""
    print("Advanced Debiased Resume Classification System Training")
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    
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
    
    # Convert sample weights to tensor
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).to(device)
    
    print(f"Class weights computed for {len(class_weights)} classes")
    print(f"Sample weights range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='models/resume_classifier_advanced_debiased',
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE * 0.8,  # Slightly lower for adversarial training
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
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
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE + 1
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
    
    print("Advanced Debiased Model Training (with adversarial debiasing)")
    print(f"Adversarial lambda: {Config.ADVERSARIAL_LAMBDA}")
    
    trainer.train()
    print("Advanced Debiased Model Training Complete")
    
    trainer.save_model('models/resume_classifier_advanced_debiased')
    tokenizer.save_pretrained('models/resume_classifier_advanced_debiased')
    print("Advanced debiased model saved")
    
    test_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    with open('results/advanced_debiased_training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nAdvanced Debiased Model Performance:")
    print(f"Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")
    
    # Also save as the main debiased model for comparison
    trainer.save_model(Config.DEBIASED_MODEL_PATH)
    tokenizer.save_pretrained(Config.DEBIASED_MODEL_PATH)
    
    return trainer, tokenizer, test_results


if __name__ == "__main__":
    train_advanced_debiased()
