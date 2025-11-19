"""
Enhanced Model Training with Improved Strategies
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import torch
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
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedResumeDataset(Dataset):
    """Enhanced PyTorch Dataset for resume classification with better error handling"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate that all inputs are proper"""
        assert len(self.texts) == len(self.labels), "Texts and labels must have same length"
        
        valid_indices = []
        for i, (text, label) in enumerate(zip(self.texts, self.labels)):
            if (isinstance(text, str) and len(text.strip()) > 10 and 
                isinstance(label, (int, np.integer)) and label >= 0):
                valid_indices.append(i)
        
        if len(valid_indices) < len(self.texts):
            print(f"Filtered {len(self.texts) - len(valid_indices)} invalid samples")
            self.texts = [self.texts[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        label = int(self.labels[idx])
        
        # Enhanced tokenization with better error handling
        try:
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
        except Exception as e:
            # Fallback for problematic texts
            print(f"Tokenization error for text: {text[:100]}... Error: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class EnhancedCustomTrainer(Trainer):
    """Enhanced trainer with advanced features"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with class weights and label smoothing"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            # Apply class weights for imbalanced data
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float).to(model.device)
            )
        else:
            # Use label smoothing for better generalization
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def enhanced_compute_metrics(eval_pred):
    """Enhanced metrics computation with detailed analysis"""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist()
    }

def enhanced_evaluate_model(trainer, test_dataset, label_map, save_report=True):
    """Comprehensive model evaluation with visualization"""
    print("\n" + "=" * 60)
    print("ENHANCED MODEL EVALUATION")
    print("=" * 60)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Calculate comprehensive metrics
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
    
    # Enhanced category analysis
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
    print(f"\nDetailed Category Performance:")
    print("-" * 50)
    sorted_categories = sorted(category_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for category, acc in sorted_categories:
        support = category_support[category]
        print(f"  {category:25s}: {acc:.4f} ({support} samples)")
    
    # Identify problematic categories
    low_performance = [(cat, acc) for cat, acc in category_accuracies.items() if acc < 0.7]
    if low_performance:
        print(f"\n⚠️  Categories Needing Improvement (<70% accuracy):")
        for cat, acc in sorted(low_performance, key=lambda x: x[1]):
            print(f"  {cat:25s}: {acc:.4f}")
    
    # Create confusion matrix visualization
    try:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(label_map))
        plt.xticks(tick_marks, [label_map[str(i)] for i in range(len(label_map))], rotation=45, ha='right')
        plt.yticks(tick_marks, [label_map[str(i)] for i in range(len(label_map))])
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i, j in np.ndindex(cm_normalized.shape):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to visualizations/confusion_matrix.png")
    except Exception as e:
        print(f"Could not create confusion matrix: {e}")
    
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
            'problematic_categories': low_performance
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/enhanced_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEnhanced evaluation results saved to results/enhanced_training_results.json")
    
    return results

def setup_enhanced_model(num_labels, model_name='roberta-base'):
    """Enhanced model setup with better configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer with better settings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with better configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        classifier_dropout=0.1,
        ignore_mismatched_sizes=True
    )
    
    model.to(device)
    print(f"Enhanced model initialized with {num_labels} classes")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device
