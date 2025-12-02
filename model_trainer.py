"""
Optimized Enhanced Model Training and Evaluation Module
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json
import os
from config import Config


class ResumeDataset(Dataset):
    """PyTorch Dataset for resume classification"""
    
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


class AdvancedFocalLoss(nn.Module):
    """Advanced Focal Loss with adaptive gamma for addressing class imbalance"""
    def __init__(self, alpha=None, gamma_range=(2, 5), reduction='mean'):
        super(AdvancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_range = gamma_range
        self.reduction = reduction
        self.current_gamma = gamma_range[0]

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Adaptive gamma based on prediction confidence
        with torch.no_grad():
            probs = F.softmax(inputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            avg_confidence = torch.mean(max_probs)
            self.current_gamma = self.gamma_range[0] + (self.gamma_range[1] - self.gamma_range[0]) * (1 - avg_confidence)
        
        focal_loss = (1 - pt) ** self.current_gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                alpha_tensor = self.alpha[targets]
                focal_loss = alpha_tensor * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTrainer(Trainer):
    """Enhanced trainer with class weighting and focal loss"""
    
    def __init__(self, class_weights=None, use_focal_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = AdvancedFocalLoss(alpha=class_weights, gamma_range=(2, 4))
        else:
            self.focal_loss = None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with class weights and optional focal loss"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.use_focal_loss and self.focal_loss:
            loss = self.focal_loss(logits, labels)
        elif self.class_weights is not None:
            device = logits.device
            if isinstance(self.class_weights, (list, tuple, np.ndarray)):
                weight_tensor = torch.tensor(self.class_weights, dtype=torch.float).to(device)
            else:
                weight_tensor = self.class_weights.to(device)
            
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits, labels)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Enhanced metrics computation with comprehensive analysis"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro and micro averages
    macro_f1 = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )[2]
    
    micro_f1 = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=0
    )[2]
    
    # Confusion matrix statistics
    cm = confusion_matrix(labels, predictions)
    diagonal = np.diag(cm)
    row_sums = cm.sum(axis=1)
    per_class_accuracy = diagonal / row_sums
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'support': support.tolist()
    }


def setup_optimized_model(num_labels, model_name):
    """Enhanced model initialization with optimized settings"""
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
        attention_probs_dropout_prob=Config.DROPOUT_RATE,
        classifier_dropout=Config.DROPOUT_RATE
    )
    
    # Initialize classification head with better initialization
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            nn.init.xavier_uniform_(model.classifier.weight)
            if model.classifier.bias is not None:
                nn.init.zeros_(model.classifier.bias)
    
    model.to(device)
    print(f"Enhanced model initialized with {num_labels} classes")
    print(f"Using dropout rate: {Config.DROPOUT_RATE}")
    
    return model, tokenizer, device


def enhanced_evaluate_model(trainer, test_dataset, label_map, save_report=True):
    """Comprehensive model evaluation with enhanced analysis"""
    print("\n" + "=" * 60)
    print("Enhanced Comprehensive Model Evaluation")
    print("=" * 60)
    
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    # Calculate macro and micro F1
    macro_f1 = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )[2]
    
    micro_f1 = precision_recall_fscore_support(
        true_labels, pred_labels, average='micro', zero_division=0
    )[2]
    
    try:
        target_names = [label_map[str(i)] for i in range(len(label_map))]
    except KeyError:
        target_names = [label_map.get(str(i), f"Category_{i}") for i in range(len(label_map))]
    
    class_report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print("\nComprehensive Category Performance Analysis:")
    print("-" * 50)
    
    category_accuracies = {}
    category_support = {}
    category_f1 = {}
    
    for i in range(len(label_map)):
        mask = true_labels == i
        if np.sum(mask) > 0:
            cat_accuracy = np.mean(pred_labels[mask] == true_labels[mask])
            cat_f1 = precision_recall_fscore_support(
                true_labels[mask], pred_labels[mask], average='weighted', zero_division=0
            )[2]
            
            try:
                category_name = label_map[str(i)]
            except KeyError:
                category_name = label_map.get(str(i), f"Category_{i}")
                
            category_accuracies[category_name] = float(cat_accuracy)
            category_support[category_name] = int(np.sum(mask))
            category_f1[category_name] = float(cat_f1)
    
    # Sort by accuracy
    sorted_by_accuracy = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Categories by Accuracy:")
    for category, acc in sorted_by_accuracy[:10]:
        support = category_support[category]
        f1_score = category_f1[category]
        print(f"  {category:25s}: Accuracy: {acc:.4f}, F1: {f1_score:.4f}, Samples: {support}")
    
    print("\nBottom 10 Categories by Accuracy:")
    for category, acc in sorted_by_accuracy[-10:]:
        support = category_support[category]
        f1_score = category_f1[category]
        print(f"  {category:25s}: Accuracy: {acc:.4f}, F1: {f1_score:.4f}, Samples: {support}")
    
    # Identify problematic categories
    problem_categories = [(cat, acc) for cat, acc in sorted_by_accuracy if acc < 0.7]
    if problem_categories:
        print(f"\nWarning: {len(problem_categories)} categories with accuracy < 70%")
        for cat, acc in problem_categories:
            print(f"  {cat:25s}: {acc:.4f}")
    
    # Calculate average accuracy for problematic categories
    if problem_categories:
        avg_problem_accuracy = np.mean([acc for _, acc in problem_categories])
        print(f"Average accuracy for problematic categories: {avg_problem_accuracy:.4f}")
    
    if save_report:
        results = {
            'eval_accuracy': float(accuracy),
            'eval_f1': float(f1),
            'eval_macro_f1': float(macro_f1),
            'eval_micro_f1': float(micro_f1),
            'eval_precision': float(precision),
            'eval_recall': float(recall),
            'per_class_accuracy': category_accuracies,
            'per_class_f1': category_f1,
            'per_class_support': category_support,
            'classification_report': class_report,
            'predictions': [int(x) for x in pred_labels.tolist()],
            'true_labels': [int(x) for x in true_labels.tolist()],
            'problem_categories': [cat for cat, acc in problem_categories],
            'problem_category_accuracies': [float(acc) for cat, acc in problem_categories]
        }
        
        # Convert numpy types in classification report
        for key, value in results['classification_report'].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'item'):
                        results['classification_report'][key][subkey] = subvalue.item()
        
        os.makedirs('results', exist_ok=True)
        with open('results/enhanced_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Enhanced evaluation results saved to results/enhanced_training_results.json")
    
    return results

def create_optimizer(model, learning_rate, weight_decay):
    """Create optimized optimizer"""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=1e-8
    )
    
    return optimizer
