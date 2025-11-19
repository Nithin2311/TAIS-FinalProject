"""
Enhanced Model training and evaluation module
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import torch
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


class EnhancedTrainer(Trainer):
    """Enhanced trainer with class weighting and gradient accumulation"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with class weights"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Apply class weights for imbalanced data
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float).to(model.device)
            )
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
    
    # Per-class metrics
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


def setup_model(num_labels, model_name):
    """Enhanced model initialization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    model.to(device)
    print(f"Model initialized with {num_labels} classes")
    
    return model, tokenizer, device


def enhanced_evaluate_model(trainer, test_dataset, label_map, save_report=True):
    """Comprehensive model evaluation with detailed analysis"""
    print("\n" + "=" * 60)
    print("ENHANCED MODEL EVALUATION")
    print("=" * 60)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    # Detailed classification report
    class_report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=[label_map[str(i)] for i in range(len(label_map))],
        output_dict=True,
        zero_division=0
    )
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Enhanced per-class accuracy analysis
    print("\nCOMPREHENSIVE CATEGORY PERFORMANCE:")
    print("-" * 50)
    
    category_accuracies = {}
    category_support = {}
    
    for i in range(len(label_map)):
        mask = true_labels == i
        if np.sum(mask) > 0:
            cat_accuracy = np.mean(pred_labels[mask] == true_labels[mask])
            category_name = label_map[str(i)]
            category_accuracies[category_name] = cat_accuracy
            category_support[category_name] = np.sum(mask)
    
    # Sort by accuracy and show all categories
    sorted_categories = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("All Categories by Accuracy:")
    for category, acc in sorted_categories:
        support = category_support[category]
        print(f"  {category:25s}: {acc:.4f} ({support} samples)")
    
    # Identify problematic categories
    problem_categories = [(cat, acc) for cat, acc in sorted_categories if acc < 0.7]
    if problem_categories:
        print(f"\n⚠️  {len(problem_categories)} categories with accuracy < 70%:")
        for cat, acc in problem_categories:
            print(f"  {cat:25s}: {acc:.4f}")
    
    # Save enhanced results
    if save_report:
        results = {
            'eval_accuracy': accuracy,
            'eval_f1': f1,
            'eval_precision': precision,
            'eval_recall': recall,
            'per_class_accuracy': category_accuracies,
            'per_class_support': category_support,
            'classification_report': class_report,
            'predictions': pred_labels.tolist(),
            'true_labels': true_labels.tolist()
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/enhanced_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEnhanced evaluation results saved to results/enhanced_training_results.json")
    
    return results
