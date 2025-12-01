"""
Baseline Model Training Script
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import pickle

from config import Config
from data_processor import download_dataset, load_and_preprocess_data, split_data
from model_trainer import ResumeDataset, EnhancedTrainer, compute_metrics, enhanced_evaluate_model, setup_optimized_model


def setup_environment():
    """Create necessary directories"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)


def save_training_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map):
    """Save training data for later bias analysis"""
    training_data = {
        'X_train': X_train,
        'X_val': X_val, 
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_map': label_map
    }
    
    with open('data/processed/training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    
    print("Training data saved for bias analysis")


def print_final_summary(test_results):
    """Print final project summary"""
    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("=" * 60)
    
    accuracy = test_results['eval_accuracy']
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    category_accuracies = test_results['per_class_accuracy']
    
    problem_categories = {cat: acc for cat, acc in category_accuracies.items() if acc < 0.7}
    improved_categories = {cat: acc for cat, acc in category_accuracies.items() if acc > 0.9}
    
    print(f"Problem categories (<70%): {len(problem_categories)}")
    print(f"Excellent categories (>90%): {len(improved_categories)}")
    
    if problem_categories:
        print("Categories needing attention:")
        for cat, acc in list(problem_categories.items())[:3]:
            print(f"    {cat}: {acc:.1%}")


def train_baseline():
    """Train baseline model"""
    print("Baseline Resume Classification System Training")
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    
    if hasattr(Config, 'display_enhanced_config'):
        Config.display_enhanced_config()
    else:
        # Fallback if method doesn't exist
        print("=" * 60)
        print("BASELINE MODEL CONFIGURATION")
        print("=" * 60)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Max Length: {Config.MAX_LENGTH} tokens")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
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
    
    with open('data/processed/enhanced_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    with open('data/processed/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    save_training_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map)
    
    model, tokenizer, device = setup_optimized_model(num_labels, Config.MODEL_NAME)
    
    train_dataset = ResumeDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = class_weights.astype(np.float32)
    print("Computed class weights")
    
    training_args = TrainingArguments(
        output_dir=Config.BASELINE_MODEL_PATH,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE,
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
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=Config.USE_FOCAL_LOSS,
        callbacks=[early_stopping]
    )
    
    print("Baseline Model Training")
    trainer.train()
    print("Baseline Model Training Complete")
    
    trainer.save_model(Config.BASELINE_MODEL_PATH)
    tokenizer.save_pretrained(Config.BASELINE_MODEL_PATH)
    print(f"Baseline model saved to {Config.BASELINE_MODEL_PATH}")
    
    test_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    with open('results/baseline_training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    with open('results/training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print_final_summary(test_results)
    
    return trainer, tokenizer, test_results


if __name__ == "__main__":
    train_baseline()
