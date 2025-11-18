"""
Model training components
CAI 6605 - Trustworthy AI Systems - Midterm Project
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import json
import os


class ResumeDataset(Dataset):
    """PyTorch Dataset for resume classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class CustomTrainer(Trainer):
    """Enhanced trainer with class weight support"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).to(self.args.device)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation with class weights"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute comprehensive metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_model(trainer, test_dataset, label_map):
    """Evaluate the trained model and save results"""
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    # Test results
    test_results = trainer.evaluate(test_dataset)
    print(f"\n🎯 Test Performance (Final):")
    print(f"  • Accuracy:  {test_results['eval_accuracy']*100:.2f}%")
    print(f"  • Precision: {test_results['eval_precision']*100:.2f}%")
    print(f"  • Recall:    {test_results['eval_recall']*100:.2f}%")
    print(f"  • F1 Score:  {test_results['eval_f1']:.4f}")
    
    # Detailed classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    print("\n" + "=" * 70)
    print("📋 PER-CATEGORY PERFORMANCE")
    print("=" * 70)
    
    category_names = [label_map[i] for i in range(len(label_map))]
    report = classification_report(
        true_labels, pred_labels,
        target_names=category_names,
        digits=3
    )
    print(report)
    
    # Top performing categories
    report_dict = classification_report(
        true_labels, pred_labels,
        target_names=category_names,
        output_dict=True
    )
    
    # Sort categories by F1 score
    category_scores = []
    for cat in category_names:
        if cat in report_dict:
            category_scores.append({
                'category': cat,
                'f1': report_dict[cat]['f1-score'],
                'support': report_dict[cat]['support']
            })
    
    category_scores.sort(key=lambda x: x['f1'], reverse=True)
    
    print("\n🏆 TOP 5 PERFORMING CATEGORIES:")
    print("-" * 50)
    for i, cat in enumerate(category_scores[:5], 1):
        print(f"{i}. {cat['category']:25s} F1: {cat['f1']:.3f} ({int(cat['support'])} samples)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'model': 'roberta-base',
        'test_accuracy': float(test_results['eval_accuracy']),
        'test_precision': float(test_results['eval_precision']),
        'test_recall': float(test_results['eval_recall']),
        'test_f1': float(test_results['eval_f1']),
        'num_categories': len(label_map),
        'top_categories': category_scores[:5]
    }
    
    with open('results/midterm_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to 'results/midterm_results.json'")
    
    return test_results


def setup_model(num_labels, model_name='roberta-base'):
    """Initialize model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    print(f"✅ Loaded {model_name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")
    
    return model, tokenizer, device
