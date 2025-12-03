"""
Debiased model training script.
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
from model_trainer import ResumeDataset, CustomTrainer, compute_metrics, evaluate_model, setup_model


def setup_environment():
    """Create necessary directories"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)


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


class DemographicSignalExtractor:
    """Extract demographic signals from anonymized resume text"""
    
    def __init__(self):
        # Gender-related keywords that might appear in anonymized resumes
        self.gender_keywords = {
            'male_leaning': [
                'football', 'basketball', 'baseball', 'hockey', 'golf', 'fishing', 'hunting',
                'military', 'marine', 'navy', 'army', 'combat', 'veteran',
                'engineering', 'mechanical', 'electrical', 'construction', 'contractor',
                'programming', 'coding', 'software', 'algorithm', 'hackathon', 'devops',
                'physics', 'mathematics', 'calculus', 'quantum', 'statistics',
                'motorcycle', 'cars', 'automotive', 'welding', 'carpentry'
            ],
            'female_leaning': [
                'nursing', 'caregiving', 'childcare', 'preschool', 'kindergarten',
                'counseling', 'therapy', 'social work', 'human resources', 'hr',
                'public relations', 'event planning', 'interior design', 'decorating',
                'early childhood', 'family therapy', 'community outreach', 'volunteer',
                'teaching', 'education', 'curriculum', 'pedagogy', 'tutoring',
                'administrative', 'receptionist', 'secretarial', 'office manager',
                'fashion', 'beauty', 'cosmetology', 'hairstyling', 'makeup'
            ]
        }
        
        # Socioeconomic/class signals
        self.class_keywords = {
            'privileged': [
                'ivy league', 'stanford', 'mit', 'harvard', 'yale', 'princeton', 'columbia',
                'private school', 'preparatory school', 'boarding school', 'prep school',
                'summer internship abroad', 'study abroad program', 'exchange program',
                'legacy', 'endowment', 'scholarship', 'fellowship', 'grant',
                'consulting', 'investment banking', 'private equity', 'venture capital',
                'executive', 'director', 'vp', 'vice president', 'c-level', 'ceo', 'cto'
            ],
            'working_class': [
                'community college', 'state university', 'public school', 'state school',
                'part-time job', 'worked through college', 'first generation',
                'evening classes', 'night shift', 'factory', 'retail', 'service industry',
                'bartender', 'waiter', 'waitress', 'cashier', 'delivery driver',
                'manual labor', 'construction worker', 'mechanic', 'plumber', 'electrician',
                'union', 'overtime', 'shift work', 'minimum wage'
            ]
        }
        
        # Industry bias patterns
        self.industry_bias = {
            'high_prestige': [
                'technology', 'software', 'engineering', 'finance', 'banking', 'consulting',
                'law', 'legal', 'medical', 'doctor', 'surgeon', 'pharmaceutical',
                'research', 'development', 'innovation', 'patent', 'intellectual property'
            ],
            'low_prestige': [
                'customer service', 'call center', 'retail', 'hospitality', 'food service',
                'cleaning', 'janitorial', 'maintenance', 'security guard', 'security officer',
                'warehouse', 'logistics', 'driver', 'delivery', 'assembly line'
            ]
        }
    
    def extract_signals(self, text):
        """Extract demographic signal features from text"""
        text_lower = text.lower()
        
        signals = {}
        
        # Gender signals
        signals['male_score'] = sum(1 for word in self.gender_keywords['male_leaning'] 
                                   if word in text_lower)
        signals['female_score'] = sum(1 for word in self.gender_keywords['female_leaning'] 
                                     if word in text_lower)
        
        # Class signals
        signals['privileged_score'] = sum(1 for word in self.class_keywords['privileged'] 
                                         if word in text_lower)
        signals['working_class_score'] = sum(1 for word in self.class_keywords['working_class'] 
                                           if word in text_lower)
        
        # Industry bias signals
        signals['high_prestige_score'] = sum(1 for word in self.industry_bias['high_prestige'] 
                                           if word in text_lower)
        signals['low_prestige_score'] = sum(1 for word in self.industry_bias['low_prestige'] 
                                          if word in text_lower)
        
        # Normalize by text length
        word_count = len(text_lower.split())
        if word_count > 0:
            for key in signals:
                signals[key] = signals[key] / word_count * 100
        
        return signals
    
    def generate_demographic_features(self, text):
        """Generate feature vector for demographic debiasing"""
        signals = self.extract_signals(text)
        
        # Convert to feature vector
        features = [
            signals.get('male_score', 0),
            signals.get('female_score', 0),
            signals.get('privileged_score', 0),
            signals.get('working_class_score', 0),
            signals.get('high_prestige_score', 0),
            signals.get('low_prestige_score', 0),
        ]
        
        # Add interaction terms
        gender_ratio = signals.get('male_score', 0) / max(1, signals.get('female_score', 1))
        class_ratio = signals.get('privileged_score', 0) / max(1, signals.get('working_class_score', 1))
        prestige_ratio = signals.get('high_prestige_score', 0) / max(1, signals.get('low_prestige_score', 1))
        
        features.extend([gender_ratio, class_ratio, prestige_ratio])
        
        return np.array(features, dtype=np.float32)


class CorrelationPenaltyLoss(nn.Module):
    """Loss function that penalizes correlation between predictions and demographic signals"""
    
    def __init__(self, lambda_corr=0.1):
        super().__init__()
        self.lambda_corr = lambda_corr
    
    def forward(self, logits, demographic_features):
        """Penalize correlation between predictions and demographic features"""
        batch_size = logits.size(0)
        
        # Get predicted probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
        
        # Normalize demographic features
        demo_norm = demographic_features - demographic_features.mean(dim=0, keepdim=True)
        demo_norm = demo_norm / (demo_norm.std(dim=0, keepdim=True) + 1e-8)
        
        # Compute correlation
        correlation = torch.matmul(probs.t(), demo_norm) / batch_size  # [num_classes, demo_dim]
        
        # Penalize high absolute correlation
        correlation_penalty = torch.mean(torch.abs(correlation))
        
        return self.lambda_corr * correlation_penalty


class CounterfactualAugmentation:
    """Counterfactual examples generation for anonymized resumes"""
    
    def __init__(self):
        # Industry swaps to create counterfactuals
        self.industry_swaps = {
            'male_leaning': {
                'software engineer': 'ux designer',
                'mechanical engineer': 'interior designer',
                'construction manager': 'event planner',
                'data scientist': 'social media manager',
                'devops engineer': 'content strategist'
            },
            'female_leaning': {
                'nurse': 'paramedic',
                'teacher': 'technical trainer',
                'hr manager': 'operations manager',
                'social worker': 'community outreach coordinator',
                'administrative assistant': 'executive assistant'
            }
        }
        
        # Keyword swaps for demographic signal modification
        self.keyword_swaps = {
            'football': 'yoga',
            'basketball': 'dance',
            'military': 'peace corps',
            'engineering': 'design',
            'coding': 'writing',
            'construction': 'coordination',
            'nursing': 'emergency response',
            'teaching': 'training',
            'childcare': 'youth development',
            'counseling': 'coaching'
        }
    
    def generate_counterfactual(self, text, swap_type='gender'):
        """Generate counterfactual version of text by swapping keywords"""
        words = text.lower().split()
        new_words = []
        
        for word in words:
            # Check if word should be swapped
            swapped = False
            for original, replacement in self.keyword_swaps.items():
                if original in word:
                    new_words.append(replacement)
                    swapped = True
                    break
            
            if not swapped:
                new_words.append(word)
        
        # Add some variation
        if random.random() < 0.3:
            # Add a random keyword from opposite category
            if swap_type == 'gender':
                male_words = ['programming', 'engineering', 'technology', 'development']
                female_words = ['communication', 'coordination', 'planning', 'organization']
                if random.random() > 0.5:
                    new_words.append(random.choice(male_words))
                else:
                    new_words.append(random.choice(female_words))
        
        return ' '.join(new_words)
    
    def augment_text(self, text, label):
        """Create augmented versions of text"""
        augmentations = []
        
        # Original
        augmentations.append((text, label))
        
        # Generate 1-2 counterfactuals
        for _ in range(random.randint(1, 2)):
            swap_type = random.choice(['gender', 'industry'])
            counterfactual = self.generate_counterfactual(text, swap_type)
            augmentations.append((counterfactual, label))
        
        return augmentations


class DebiasingDataset(ResumeDataset):
    """Dataset with demographic features for debiasing"""
    
    def __init__(self, texts, labels, demographic_features, tokenizer, max_length):
        super().__init__(texts, labels, tokenizer, max_length)
        self.demographic_features = demographic_features
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # Add demographic features
        demo_feat = self.demographic_features[idx]
        item['demographic_features'] = torch.tensor(demo_feat, dtype=torch.float32)
        
        return item


class BiasMitigationPipeline:
    """Bias mitigation with targeted augmentation and reweighting"""
    
    def __init__(self):
        self.signal_extractor = DemographicSignalExtractor()
        self.counterfactual_aug = CounterfactualAugmentation()
    
    def apply_debiasing(self, X_train, y_train, X_val, y_val):
        """Apply debiasing techniques"""
        
        print("Extracting demographic signals...")
        
        # Extract demographic signals
        demo_features_train = []
        for text in X_train:
            features = self.signal_extractor.generate_demographic_features(text)
            demo_features_train.append(features)
        
        # Identify underrepresented groups
        underrepresented_indices = self._identify_underrepresented_groups(
            demo_features_train, y_train
        )
        
        # Apply targeted augmentation
        print(f"Applying targeted augmentation to {len(underrepresented_indices)} samples...")
        X_augmented, y_augmented, demo_augmented = self._apply_targeted_augmentation(
            X_train, y_train, demo_features_train, underrepresented_indices
        )
        
        # Compute sample weights
        sample_weights = self._compute_fairness_weights(
            y_augmented, demo_augmented
        )
        
        print(f"Augmented dataset: {len(X_train)} -> {len(X_augmented)} samples")
        
        return X_augmented, y_augmented, sample_weights, demo_augmented
    
    def _identify_underrepresented_groups(self, demo_features, labels):
        """Identify samples from underrepresented demographic groups"""
        underrepresented = []
        
        # Convert to numpy for easier processing
        demo_np = np.array(demo_features)
        
        # Identify samples with extreme demographic signals
        # High gender bias
        gender_bias = np.abs(demo_np[:, 0] - demo_np[:, 1])
        high_gender_bias = np.where(gender_bias > np.percentile(gender_bias, 75))[0]
        
        # High class bias
        class_bias = np.abs(demo_np[:, 2] - demo_np[:, 3])
        high_class_bias = np.where(class_bias > np.percentile(class_bias, 75))[0]
        
        underrepresented.extend(high_gender_bias.tolist())
        underrepresented.extend(high_class_bias.tolist())
        
        # Add rare labels
        label_counts = Counter(labels)
        avg_count = np.mean(list(label_counts.values()))
        for label, count in label_counts.items():
            if count < avg_count * 0.5:  # Less than half the average
                label_indices = [i for i, l in enumerate(labels) if l == label]
                underrepresented.extend(label_indices[:min(5, len(label_indices))])
        
        return list(set(underrepresented))
    
    def _apply_targeted_augmentation(self, texts, labels, demo_features, target_indices):
        """Apply augmentation to targeted samples"""
        X_augmented = list(texts)
        y_augmented = list(labels)
        demo_augmented = list(demo_features)
        
        for idx in target_indices:
            if idx < len(texts):
                text = texts[idx]
                label = labels[idx]
                demo_feat = demo_features[idx]
                
                # Generate 1-3 augmentations per target sample
                for _ in range(random.randint(1, 3)):
                    augmented_text = self.counterfactual_aug.generate_counterfactual(
                        text, random.choice(['gender', 'industry'])
                    )
                    X_augmented.append(augmented_text)
                    y_augmented.append(label)
                    
                    # Slightly modify demographic features for augmented sample
                    modified_demo = demo_feat.copy()
                    noise = np.random.normal(0, 0.1, size=modified_demo.shape)
                    modified_demo = modified_demo + noise
                    demo_augmented.append(modified_demo)
        
        return X_augmented, y_augmented, demo_augmented
    
    def _compute_fairness_weights(self, labels, demo_features):
        """Compute sample weights to encourage fairness"""
        # Base weights from class distribution
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        base_weights = []
        for label in labels:
            weight = total_samples / (len(label_counts) * max(label_counts[label], 1))
            base_weights.append(min(weight, 3.0))  # Cap at 3x
        
        base_weights = np.array(base_weights, dtype=np.float32)
        
        # Adjust weights based on demographic features
        demo_np = np.array(demo_features)
        
        # Penalize samples with extreme demographic bias
        gender_bias = np.abs(demo_np[:, 0] - demo_np[:, 1])
        class_bias = np.abs(demo_np[:, 2] - demo_np[:, 3])
        
        # Normalize biases
        gender_norm = (gender_bias - np.mean(gender_bias)) / (np.std(gender_bias) + 1e-8)
        class_norm = (class_bias - np.mean(class_bias)) / (np.std(class_bias) + 1e-8)
        
        # Higher weight for balanced samples
        fairness_multiplier = 1.0 / (1.0 + np.abs(gender_norm) + np.abs(class_norm))
        
        # Combine weights
        final_weights = base_weights * fairness_multiplier
        
        # Normalize
        final_weights = final_weights / np.mean(final_weights)
        
        return final_weights


class FairnessAwareTrainer(CustomTrainer):
    """Trainer with fairness regularization"""
    
    def __init__(self, correlation_lambda=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_lambda = correlation_lambda
        self.correlation_loss = CorrelationPenaltyLoss(lambda_corr=correlation_lambda)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with fairness regularization"""
        labels = inputs.get("labels")
        demographic_features = inputs.get("demographic_features")
        
        # Remove demographic features from inputs to model
        model_inputs = {k: v for k, v in inputs.items() 
                       if k not in ['demographic_features']}
        
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        
        # Main classification loss (from parent)
        main_loss = super().compute_loss(model, model_inputs, return_outputs=False, **kwargs)
        
        # Fairness regularization loss
        if demographic_features is not None:
            fairness_loss = self.correlation_loss(logits, demographic_features)
            total_loss = main_loss + fairness_loss
        else:
            total_loss = main_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def train_debiased():
    """Train debiased model with bias mitigation techniques"""
    print("Debiased Resume Classification System Training")
    print("=" * 60)
    
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
    
    # Convert to lists
    X_train = list(X_train)
    y_train = list(y_train)
    X_val = list(X_val)
    y_val = list(y_val)
    X_test = list(X_test)
    y_test = list(y_test)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Apply bias mitigation
    pipeline = BiasMitigationPipeline()
    X_train_debiased, y_train_debiased, sample_weights, demo_features = pipeline.apply_debiasing(
        X_train, y_train, X_val, y_val
    )
    
    print(f"\nAfter debiasing:")
    print(f"  Train: {len(X_train_debiased)} samples")
    
    # Setup model
    model, tokenizer, device = setup_model(num_labels, Config.MODEL_NAME)
    
    # Create datasets
    train_dataset = DebiasingDataset(
        X_train_debiased, y_train_debiased, demo_features, tokenizer, Config.MAX_LENGTH
    )
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_debiased),
        y=y_train_debiased
    )
    class_weights = class_weights.astype(np.float32)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.DEBIASED_MODEL_PATH,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE * 0.8,  # Slightly lower for stability
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
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
    
    # Create trainer
    trainer = FairnessAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=Config.USE_FOCAL_LOSS,
        correlation_lambda=Config.ADVERSARIAL_LAMBDA,
        callbacks=[early_stopping]
    )
    
    print(f"\nTraining configuration:")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Fairness lambda: {Config.ADVERSARIAL_LAMBDA}")
    
    # Train
    print(f"\nStarting training...")
    trainer.train()
    print("Training complete")
    
    # Save model
    trainer.save_model(Config.DEBIASED_MODEL_PATH)
    tokenizer.save_pretrained(Config.DEBIASED_MODEL_PATH)
    
    # Save label map
    with open(f'{Config.DEBIASED_MODEL_PATH}/label_map.json', 'w') as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    
    print(f"\nModel saved to {Config.DEBIASED_MODEL_PATH}")
    
    # Evaluate
    test_results = evaluate_model(trainer, test_dataset, label_map)
    
    test_results_serializable = convert_numpy_types(test_results)
    
    # Save results
    results_path = 'results/debiased_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    print(f"\nDebiased Model Performance:")
    print(f"  Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    print(f"  F1 Score: {test_results['eval_f1']:.4f}")
    print(f"  Macro F1 Score: {test_results['eval_macro_f1']:.4f}")
    
    # Save training summary
    summary = {
        'model_type': 'debiased',
        'dataset_size': {
            'original_train': len(X_train),
            'augmented_train': len(X_train_debiased),
            'val': len(X_val),
            'test': len(X_test)
        },
        'performance': {
            'accuracy': float(test_results['eval_accuracy']),
            'f1': float(test_results['eval_f1']),
            'macro_f1': float(test_results['eval_macro_f1'])
        },
        'config': {
            'model_name': Config.MODEL_NAME,
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': training_args.learning_rate,
            'fairness_lambda': Config.ADVERSARIAL_LAMBDA,
            'num_epochs': Config.NUM_EPOCHS
        }
    }
    
    with open('results/debiased_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to results/debiased_training_summary.json")
    
    return trainer, tokenizer, test_results_serializable


if __name__ == "__main__":
    train_debiased()
