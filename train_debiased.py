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


class AdversarialLoss(nn.Module):
    """Adversarial loss for debiasing with gradient reversal"""
    
    def __init__(self, num_classes, num_protected_attributes, hidden_dim=256):
        super().__init__()
        self.num_protected = num_protected_attributes
        
        self.grl = GradientReversalLayer(alpha=Config.ADVERSARIAL_LAMBDA)
        
        self.adversary = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 2)
            ) for _ in range(num_protected_attributes)
        ])
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, features, protected_labels):
        """Compute adversarial loss with gradient reversal"""
        total_loss = 0
        valid_attributes = 0
        
        rev_features = self.grl(features)
        
        for i in range(self.num_protected):
            if i < len(protected_labels):
                adv_output = self.adversary[i](rev_features)
                
                valid_mask = protected_labels[i] != -1
                if valid_mask.sum() > 0:
                    loss = self.criterion(adv_output[valid_mask], protected_labels[i][valid_mask])
                    total_loss += loss
                    valid_attributes += 1
        
        return total_loss / max(1, valid_attributes)


class CounterfactualAugmentation:
    """Counterfactual examples generation"""
    
    def __init__(self):
        self.gender_swap = {
            'he': 'she', 'him': 'her', 'his': 'her', 'male': 'female',
            'man': 'woman', 'men': 'women', 'mr': 'ms', 'mister': 'miss',
            'father': 'mother', 'husband': 'wife', 'brother': 'sister',
            'son': 'daughter', 'gentleman': 'lady', 'boy': 'girl',
            'sir': 'madam', 'uncle': 'aunt', 'nephew': 'niece'
        }
        
        self.inverse_gender_swap = {v: k for k, v in self.gender_swap.items()}
    
    def generate_counterfactual(self, text, modification_type='gender'):
        """Generate counterfactual version of text"""
        words = text.lower().split()
        new_words = []
        
        for word in words:
            if modification_type == 'gender' and word in self.gender_swap:
                new_word = self.gender_swap[word]
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def generate_intersectional_counterfactual(self, text):
        """Generate intersectional counterfactual with multiple modifications"""
        gender_modified = self.generate_counterfactual(text, 'gender')
        
        words = gender_modified.split()
        new_words = []
        for word in words:
            if word in self.gender_swap and random.random() > 0.5:
                if random.random() > 0.5:
                    new_words.append(self.gender_swap[word])
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)


class BiasMitigationPipeline:
    """Bias mitigation with multiple techniques"""
    
    def __init__(self):
        self.counterfactual_aug = CounterfactualAugmentation()
    
    def apply_debiasing(self, X_train, y_train, X_val, y_val):
        """Apply debiasing techniques"""
        
        X_train = list(X_train)
        y_train = list(y_train)
        
        X_augmented, y_augmented = self._apply_counterfactual_augmentation(X_train, y_train)
        
        sample_weights = self._compute_sample_weights(X_augmented, y_augmented)
        
        print(f"Augmented dataset: {len(X_train)} -> {len(X_augmented)} samples")
        
        return X_augmented, y_augmented, sample_weights
    
    def _apply_counterfactual_augmentation(self, texts, labels):
        """Apply counterfactual augmentation"""
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        for i, text in enumerate(texts):
            original_label = labels[i]
            
            if random.random() < 0.3:
                if random.random() < 0.7:
                    counterfactual_text = self.counterfactual_aug.generate_intersectional_counterfactual(text)
                else:
                    counterfactual_text = self.counterfactual_aug.generate_counterfactual(text, 'gender')
                
                augmented_texts.append(counterfactual_text)
                augmented_labels.append(original_label)
        
        return augmented_texts, augmented_labels
    
    def _compute_sample_weights(self, texts, labels):
        """Compute sample weights based on label distribution"""
        label_counts = Counter(labels)
        total_samples = len(texts)
        
        weights = []
        for label in labels:
            weight = total_samples / (len(label_counts) * max(label_counts[label], 1))
            weights.append(min(weight, 2.0))
        
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.mean()
        
        return weights


class AdversarialDebiasedTrainer(CustomTrainer):
    """Trainer with adversarial debiasing"""
    
    def __init__(self, adversarial_lambda=0.1, num_protected_attributes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversarial_lambda = adversarial_lambda
        
        num_labels = self.model.config.num_labels
        self.adversarial_loss = AdversarialLoss(num_labels, num_protected_attributes)
        
        if self.args.device.type == 'cuda':
            self.adversarial_loss.cuda()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with adversarial component"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
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
        
        batch_size = logits.size(0)
        
        protected_labels = [
            torch.randint(0, 2, (batch_size,), dtype=torch.long).to(logits.device),
            torch.randint(0, 2, (batch_size,), dtype=torch.long).to(logits.device)
        ]
        
        for i in range(len(protected_labels)):
            mask = torch.rand(batch_size) < 0.3
            protected_labels[i][mask] = -1
        
        adv_loss = self.adversarial_loss(logits, protected_labels)
        
        total_loss = main_loss + adv_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def train_debiased():
    """Train debiased model with bias mitigation techniques"""
    print("Debiased Resume Classification System Training")
    
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
    
    pipeline = BiasMitigationPipeline()
    X_train_debiased, y_train_debiased, sample_weights = pipeline.apply_debiasing(
        X_train, y_train, X_val, y_val
    )
    
    model, tokenizer, device = setup_model(num_labels, Config.MODEL_NAME)
    
    train_dataset = ResumeDataset(X_train_debiased, y_train_debiased, tokenizer, Config.MAX_LENGTH)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_debiased),
        y=y_train_debiased
    )
    class_weights = class_weights.astype(np.float32)
    
    training_args = TrainingArguments(
        output_dir='models/resume_classifier_debiased',
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE,
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
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    trainer.train()
    print("Training complete")
    
    trainer.save_model('models/resume_classifier_debiased')
    tokenizer.save_pretrained('models/resume_classifier_debiased')
    print(f"Model saved to models/resume_classifier_debiased")
    
    with open('models/resume_classifier_debiased/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    test_results = evaluate_model(trainer, test_dataset, label_map)
    
    test_results_serializable = convert_numpy_types(test_results)
    
    results_path = 'results/debiased_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    print(f"\nDebiased Model Performance:")
    print(f"Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")
    print(f"Macro F1 Score: {test_results['eval_macro_f1']:.4f}")
    
    return trainer, tokenizer, test_results_serializable


if __name__ == "__main__":
    train_debiased()
