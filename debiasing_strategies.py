"""
Bias Mitigation Strategies Implementation
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import re


class PreprocessingDebiasing:
    """Pre-processing bias mitigation techniques"""
    
    def __init__(self):
        self.demographic_keywords = {
            'gender': ['he', 'she', 'him', 'her', 'his', 'hers', 'male', 'female', 'man', 'woman'],
            'race': ['african', 'asian', 'hispanic', 'caucasian', 'white', 'black', 'ethnicity'],
            'privilege': ['ivy league', 'legacy', 'first generation', 'low income', 'underserved']
        }
    
    def remove_demographic_indicators(self, text):
        """Remove explicit demographic indicators from text"""
        cleaned_text = text.lower()
        
        # Remove gender indicators
        for keyword in self.demographic_keywords['gender']:
            cleaned_text = re.sub(r'\b' + keyword + r'\b', '[REDACTED]', cleaned_text)
        
        # Remove other sensitive indicators
        for keyword in self.demographic_keywords['race'] + self.demographic_keywords['privilege']:
            cleaned_text = re.sub(r'\b' + keyword + r'\b', '[REDACTED]', cleaned_text)
        
        return cleaned_text
    
    def balance_dataset(self, texts, labels, demographics, target_attribute):
        """Balance dataset across demographic groups"""
        # Group by demographic attribute and label
        groups = {}
        for i, (text, label, demo) in enumerate(zip(texts, labels, demographics[target_attribute])):
            key = (label, demo)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Find maximum size per group
        max_size = max(len(indices) for indices in groups.values())
        
        # Resample each group to maximum size
        balanced_indices = []
        for indices in groups.values():
            if len(indices) < max_size:
                # Oversample minority groups
                resampled_indices = resample(indices, replace=True, n_samples=max_size, random_state=42)
            else:
                resampled_indices = indices
            balanced_indices.extend(resampled_indices)
        
        balanced_texts = [texts[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        return balanced_texts, balanced_labels


class AdversarialDebiasing(nn.Module):
    """In-processing adversarial debiasing"""
    
    def __init__(self, main_model, num_classes, num_demographics):
        super(AdversarialDebiasing, self).__init__()
        self.main_model = main_model
        self.num_classes = num_classes
        self.num_demographics = num_demographics
        
        # Adversarial classifier to predict protected attribute
        self.adversary = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_demographics)
        )
    
    def forward(self, input_ids, attention_mask, adversary_lambda=0.1):
        # Main task forward pass
        main_outputs = self.main_model(input_ids=input_ids, attention_mask=attention_mask)
        main_logits = main_outputs.logits
        
        # Adversarial pass - try to predict protected attribute from main task features
        adversary_logits = self.adversary(main_logits.detach())  # Detach to not affect main model
        
        return main_logits, adversary_logits
    
    def adversarial_loss(self, main_logits, adversary_logits, main_labels, protected_labels, lambda_val=0.1):
        """Compute combined main task and adversarial loss"""
        # Main task loss
        main_loss = nn.CrossEntropyLoss()(main_logits, main_labels)
        
        # Adversarial loss (we want adversary to fail)
        adversary_loss = nn.CrossEntropyLoss()(adversary_logits, protected_labels)
        
        # Combined loss: maximize main performance while minimizing adversary performance
        total_loss = main_loss - lambda_val * adversary_loss
        
        return total_loss


class PostprocessingDebiasing:
    """Post-processing bias mitigation techniques"""
    
    def __init__(self, label_map):
        self.label_map = label_map
    
    def demographic_parity_calibration(self, probabilities, demographics, protected_group, threshold_adjustment=0.1):
        """Adjust decision thresholds to achieve demographic parity"""
        calibrated_predictions = []
        
        for i, probs in enumerate(probabilities):
            demo = demographics[i]
            if demo == protected_group:
                # Adjust threshold for protected group
                adjusted_probs = probs * (1 + threshold_adjustment)
            else:
                adjusted_probs = probs
            
            predicted_class = np.argmax(adjusted_probs)
            calibrated_predictions.append(predicted_class)
        
        return calibrated_predictions
    
    def equalized_odds_calibration(self, probabilities, true_labels, demographics, protected_group):
        """Calibrate predictions to achieve equalized odds"""
        # Calculate error rates by demographic group
        group_error_rates = {}
        demo_groups = np.unique(demographics)
        
        for group in demo_groups:
            group_mask = demographics == group
            if np.sum(group_mask) > 0:
                group_predictions = np.argmax(probabilities[group_mask], axis=1)
                group_accuracy = np.mean(group_predictions == true_labels[group_mask])
                group_error_rates[group] = 1 - group_accuracy
        
        # Find maximum error rate
        max_error = max(group_error_rates.values())
        
        # Adjust probabilities for groups with lower error rates
        calibrated_predictions = []
        for i, probs in enumerate(probabilities):
            group = demographics[i]
            group_error = group_error_rates[group]
            
            if group_error < max_error:
                # Add noise to reduce accuracy for over-performing group
                noise = np.random.normal(0, (max_error - group_error) * 0.1, len(probs))
                adjusted_probs = probs + noise
            else:
                adjusted_probs = probs
            
            predicted_class = np.argmax(adjusted_probs)
            calibrated_predictions.append(predicted_class)
        
        return calibrated_predictions


class BiasMitigationPipeline:
    """Orchestrate multiple bias mitigation strategies"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        
        self.preprocessor = PreprocessingDebiasing()
        self.postprocessor = PostprocessingDebiasing(label_map)
    
    def apply_preprocessing_debiasing(self, texts, labels, demographics):
        """Apply pre-processing debiasing techniques"""
        print("🔄 Applying pre-processing debiasing...")
        
        # Remove demographic indicators
        debiased_texts = [self.preprocessor.remove_demographic_indicators(text) for text in texts]
        
        # Balance dataset for gender
        balanced_texts, balanced_labels = self.preprocessor.balance_dataset(
            debiased_texts, labels, demographics, 'gender'
        )
        
        return balanced_texts, balanced_labels
    
    def apply_inprocessing_debiasing(self, train_dataset, val_dataset, demographics):
        """Apply in-processing adversarial debiasing"""
        print("🔄 Applying in-processing adversarial debiasing...")
        
        # Convert demographics to numerical labels
        demo_encoder = LabelEncoder()
        protected_labels = demo_encoder.fit_transform(demographics['gender'])
        
        # Initialize adversarial model
        num_classes = len(self.label_map)
        num_demographics = len(demo_encoder.classes_)
        
        adversarial_model = AdversarialDebiasing(
            self.model, num_classes, num_demographics
        ).to(self.device)
        
        return adversarial_model, protected_labels
    
    def apply_postprocessing_debiasing(self, probabilities, true_labels, demographics):
        """Apply post-processing debiasing techniques"""
        print("🔄 Applying post-processing debiasing...")
        
        # Convert demographics to array
        gender_demographics = np.array(demographics['gender'])
        
        # Apply demographic parity calibration
        calibrated_predictions = self.postprocessor.demographic_parity_calibration(
            probabilities, gender_demographics, 'female'
        )
        
        return calibrated_predictions
    
    def comprehensive_debiasing(self, train_texts, train_labels, val_texts, val_labels, demographics):
        """Apply comprehensive debiasing pipeline"""
        print("\n" + "=" * 70)
        print("🛡️  COMPREHENSIVE BIAS MITIGATION PIPELINE")
        print("=" * 70)
        
        # 1. Pre-processing debiasing
        debiased_train_texts, debiased_train_labels = self.apply_preprocessing_debiasing(
            train_texts, train_labels, demographics
        )
        
        print(f"✅ Pre-processing complete: {len(debiased_train_texts)} samples")
        
        # Note: In-processing would be applied during training
        # Note: Post-processing would be applied during inference
        
        return debiased_train_texts, debiased_train_labels
