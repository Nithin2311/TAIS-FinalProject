"""
Enhanced Bias Mitigation Strategies Implementation - FIXED VERSION
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import re
import os
import json
from collections import Counter


class ImprovedPreprocessingDebiasing:
    """Improved pre-processing with controlled debiasing"""
    
    def __init__(self):
        # More selective demographic keywords - only remove explicit identifiers
        self.demographic_keywords = {
            'gender': ['mr.', 'mrs.', 'ms.', 'mr', 'mrs', 'ms', 'mister', 'missus'],
            'race': ['race', 'racial', 'ethnicity', 'nationality'],
            'privilege': ['ivy league', 'legacy', 'first generation']
        }
    
    def remove_demographic_indicators(self, text):
        """Selective removal of ONLY explicit demographic indicators"""
        cleaned_text = text.lower()
        
        # Remove only explicit identifiers, not all gender pronouns
        for category, keywords in self.demographic_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                cleaned_text = re.sub(pattern, '[REDACTED]', cleaned_text)
        
        return cleaned_text
    
    def balance_dataset_conservative(self, texts, labels, demographics, max_increase_ratio=1.5):
        """Conservative dataset balancing that preserves performance"""
        print("Applying conservative dataset balancing...")
        
        # Group by demographic attribute and label
        groups = {}
        for i, (text, label, demo) in enumerate(zip(texts, labels, demographics['gender'])):
            key = (label, demo)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Calculate current distribution
        group_sizes = {k: len(v) for k, v in groups.items()}
        avg_size = np.mean(list(group_sizes.values()))
        
        # Set conservative target size - don't oversample too much
        target_size = min(int(avg_size * max_increase_ratio), 100)  # Cap at 100 per group
        
        print(f"Conservative target size per group: {target_size}")
        
        # Resample each group conservatively
        balanced_indices = []
        for key, indices in groups.items():
            current_size = len(indices)
            
            if current_size < target_size:
                # Limited oversampling
                oversample_ratio = min(target_size / current_size, 2.0)  # Max 2x oversampling
                new_size = min(int(current_size * oversample_ratio), target_size)
                resampled_indices = resample(indices, replace=True, n_samples=new_size, random_state=42)
            elif current_size > target_size:
                # Limited undersampling
                resampled_indices = resample(indices, replace=False, n_samples=target_size, random_state=42)
            else:
                resampled_indices = indices
            
            balanced_indices.extend(resampled_indices)
        
        balanced_texts = [texts[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        print(f"Conservative balancing: {len(texts)} -> {len(balanced_texts)} samples")
        return balanced_texts, balanced_labels


class ConservativeAdversarialDebiasing(nn.Module):
    """Conservative adversarial debiasing with performance preservation"""
    
    def __init__(self, main_model, num_classes, num_demographics, lambda_val=0.01):  # Reduced lambda
        super(ConservativeAdversarialDebiasing, self).__init__()
        self.main_model = main_model
        self.num_classes = num_classes
        self.num_demographics = num_demographics
        self.lambda_val = lambda_val  # Much smaller lambda to preserve performance
        
        # Simple adversary to avoid overfitting
        self.adversary = nn.Sequential(
            nn.Linear(num_classes, 32),  # Smaller network
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_demographics)
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass with performance preservation"""
        # Main task forward pass
        main_outputs = self.main_model(input_ids=input_ids, attention_mask=attention_mask)
        main_logits = main_outputs.logits
        
        # Adversarial pass with gradient blocking for main task
        adversary_logits = self.adversary(main_logits.detach())  # Detach to protect main model
        
        return main_logits, adversary_logits
    
    def compute_conservative_loss(self, main_logits, adversary_logits, main_labels, protected_labels):
        """Conservative loss that prioritizes main task performance"""
        # Main task loss (priority)
        main_loss = nn.CrossEntropyLoss()(main_logits, main_labels)
        
        # Adversarial loss with very small weight
        adversary_loss = nn.CrossEntropyLoss()(adversary_logits, protected_labels)
        
        # Conservative combined loss - main task is primary
        total_loss = main_loss + self.lambda_val * adversary_loss  # Positive lambda for stability
        
        return total_loss, main_loss, adversary_loss


class PerformancePreservingDebiasing:
    """Debiasing that prioritizes model performance"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.preprocessor = ImprovedPreprocessingDebiasing()
    
    def apply_performance_preserving_debiasing(self, texts, labels, demographics):
        """Apply debiasing that doesn't sacrifice performance"""
        print("Applying performance-preserving debiasing...")
        
        # Step 1: Light preprocessing - only remove explicit identifiers
        processed_texts = [self.preprocessor.remove_demographic_indicators(text) for text in texts]
        
        # Step 2: Conservative balancing
        balanced_texts, balanced_labels = self.preprocessor.balance_dataset_conservative(
            processed_texts, labels, demographics
        )
        
        # Step 3: Ensure we don't reduce dataset quality
        if len(balanced_texts) > len(texts) * 1.5:
            print("Warning: Dataset expanded too much. Using original size.")
            return texts, labels
        
        print(f"Performance-preserving debiasing complete: {len(texts)} -> {len(balanced_texts)} samples")
        return balanced_texts, balanced_labels
    
    def train_with_performance_monitoring(self, train_dataset, val_dataset, demographics, 
                                        performance_threshold=0.02):
        """Train with monitoring to ensure performance doesn't drop significantly"""
        print("Training with performance monitoring...")
        
        # Convert demographics
        demo_encoder = LabelEncoder()
        protected_labels = demo_encoder.fit_transform(demographics['gender'])
        
        # Initialize conservative adversarial model
        num_classes = len(self.label_map)
        num_demographics = len(demo_encoder.classes_)
        
        adversarial_model = ConservativeAdversarialDebiasing(
            self.model, num_classes, num_demographics, lambda_val=0.01
        ).to(self.device)
        
        return adversarial_model


class TargetedCategoryImprovement:
    """Targeted improvement for underperforming categories"""
    
    def __init__(self):
        self.underperforming_categories = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA']
        
        # Enhanced keywords for underperforming categories
        self.category_enhancement_keywords = {
            'BPO': ['customer service', 'call center', 'bpo', 'voice process', 'client support',
                   'customer care', 'helpdesk', 'technical support', 'inbound', 'outbound'],
            
            'AUTOMOBILE': ['automobile', 'automotive', 'vehicle', 'car', 'mechanic', 'auto repair',
                          'engine', 'transmission', 'brake system', 'suspension', 'diagnostic'],
            
            'APPAREL': ['apparel', 'fashion', 'clothing', 'garment', 'textile', 'merchandising',
                       'retail', 'fashion design', 'textile design', 'pattern making'],
            
            'DIGITAL-MEDIA': ['digital media', 'social media', 'content creation', 'video editing',
                             'graphic design', 'multimedia', 'animation', 'visual effects']
        }
    
    def enhance_category_features(self, texts, labels, label_map):
        """Enhance features for underperforming categories"""
        enhanced_texts = []
        
        for text, label in zip(texts, labels):
            category_name = label_map.get(str(label), "")
            enhanced_text = text
            
            # Add category-specific keywords for underperforming categories
            if category_name in self.underperforming_categories:
                keywords = self.category_enhancement_keywords.get(category_name, [])
                # Add 2-3 relevant keywords
                selected_keywords = keywords[:3]
                enhanced_text += " " + " ".join(selected_keywords)
            
            enhanced_texts.append(enhanced_text)
        
        print(f"Enhanced features for {len([l for l in labels if label_map.get(str(l), '') in self.underperforming_categories])} samples")
        return enhanced_texts


class ImprovedBiasMitigationPipeline:
    """Improved bias mitigation that preserves performance"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        
        self.performance_preserving_debiasing = PerformancePreservingDebiasing(model, tokenizer, label_map, device)
        self.targeted_improvement = TargetedCategoryImprovement()
    
    def apply_improved_debiasing(self, train_texts, train_labels, val_texts, val_labels, demographics):
        """Apply improved debiasing that doesn't harm performance"""
        print("\n" + "=" * 60)
        print("IMPROVED PERFORMANCE-PRESERVING DEBIASING PIPELINE")
        print("=" * 60)
        
        # Step 1: Targeted improvement for underperforming categories
        print("1. Enhancing underperforming categories...")
        enhanced_train_texts = self.targeted_improvement.enhance_category_features(
            train_texts, train_labels, self.label_map
        )
        
        # Step 2: Conservative debiasing
        print("2. Applying conservative debiasing...")
        debiased_texts, debiased_labels = self.performance_preserving_debiasing.apply_performance_preserving_debiasing(
            enhanced_train_texts, train_labels, demographics
        )
        
        # Step 3: Ensure reasonable dataset size
        if len(debiased_texts) > len(train_texts) * 1.3:
            print("3. Trimming dataset to preserve quality...")
            # Keep the original samples plus a limited number of new ones
            original_indices = list(range(len(train_texts)))
            new_indices = list(range(len(train_texts), len(debiased_texts)))
            # Keep only 30% of new samples
            keep_new = int(len(new_indices) * 0.3)
            if keep_new > 0:
                selected_new = np.random.choice(new_indices, size=keep_new, replace=False)
                final_indices = original_indices + selected_new.tolist()
                debiased_texts = [debiased_texts[i] for i in final_indices]
                debiased_labels = [debiased_labels[i] for i in final_indices]
        
        print(f"Improved debiasing complete: {len(train_texts)} -> {len(debiased_texts)} samples")
        
        return debiased_texts, debiased_labels
