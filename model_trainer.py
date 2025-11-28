"""
Enhanced Model training and evaluation module
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import os
from config import Config


class ResumeDataset(Dataset):
    """Enhanced PyTorch Dataset for resume classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTrainer(Trainer):
    """Enhanced trainer with class weighting, focal loss, and gradient accumulation"""
    
    def __init__(self, class_weights=None, use_focal_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_loss = FocalLoss(gamma=2.0) if use_focal_loss else None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with class weights and optional focal loss"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.use_focal_loss and self.focal_loss:
            loss = self.focal_loss(logits, labels)
        elif self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float).to(model.device)
            )
            loss = loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Enhanced metrics computation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist()
    }


def setup_optimized_model(num_labels, model_name):
    """Enhanced model initialization with better configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=Config.DROPOUT_RATE,
        attention_probs_dropout_prob=Config.DROPOUT_RATE
    )
    
    model.to(device)
    print(f"Enhanced model initialized with {num_labels} classes")
    print(f"Using dropout rate: {Config.DROPOUT_RATE}")
    
    return model, tokenizer, device


def enhanced_evaluate_model(trainer, test_dataset, label_map, save_report=True):
    """Comprehensive model evaluation with detailed analysis"""
    print("\n" + "=" * 60)
    print("ENHANCED MODEL EVALUATION")
    print("=" * 60)
    
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    try:
        target_names = [label_map[i] for i in range(len(label_map))]
    except KeyError:
        target_names = [label_map[str(i)] for i in range(len(label_map))]
    
    class_report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("\nCOMPREHENSIVE CATEGORY PERFORMANCE:")
    print("-" * 50)
    
    category_accuracies = {}
    category_support = {}
    
    for i in range(len(label_map)):
        mask = true_labels == i
        if np.sum(mask) > 0:
            cat_accuracy = np.mean(pred_labels[mask] == true_labels[mask])
            try:
                category_name = label_map[i]
            except KeyError:
                category_name = label_map[str(i)]
            category_accuracies[category_name] = float(cat_accuracy)
            category_support[category_name] = int(np.sum(mask))
    
    sorted_categories = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("All Categories by Accuracy:")
    for category, acc in sorted_categories:
        support = category_support[category]
        print(f"  {category:25s}: {acc:.4f} ({support} samples)")
    
    problem_categories = [(cat, acc) for cat, acc in sorted_categories if acc < 0.7]
    if problem_categories:
        print(f"\nWARNING: {len(problem_categories)} categories with accuracy < 70%:")
        for cat, acc in problem_categories:
            print(f"  {cat:25s}: {acc:.4f}")
    
    if save_report:
        results = {
            'eval_accuracy': float(accuracy),
            'eval_f1': float(f1),
            'eval_precision': float(precision),
            'eval_recall': float(recall),
            'per_class_accuracy': category_accuracies,
            'per_class_support': category_support,
            'classification_report': class_report,
            'predictions': [int(x) for x in pred_labels.tolist()],
            'true_labels': [int(x) for x in true_labels.tolist()]
        }
        
        for key, value in results['classification_report'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'item'):
                        results['classification_report'][key][subkey] = subvalue.item()
        
        os.makedirs('results', exist_ok=True)
        with open('results/enhanced_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEnhanced evaluation results saved to results/enhanced_training_results.json")
    
    return results
