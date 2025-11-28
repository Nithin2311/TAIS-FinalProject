"""
Enhanced Training script for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import os
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import download_dataset, load_and_preprocess_data, split_data
from model_trainer import ResumeDataset, EnhancedTrainer, compute_metrics, enhanced_evaluate_model, setup_optimized_model
import torch
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import pickle


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
    print("FINAL PROJECT COMPLETION SUMMARY")
    print("=" * 60)
    
    accuracy = test_results['eval_accuracy']
    print(f"FINAL TEST ACCURACY: {accuracy*100:.2f}%")
    
    category_accuracies = test_results['per_class_accuracy']
    
    problem_categories = {cat: acc for cat, acc in category_accuracies.items() if acc < 0.7}
    improved_categories = {cat: acc for cat, acc in category_accuracies.items() if acc > 0.9}
    
    print(f"\nCATEGORY PERFORMANCE ANALYSIS:")
    print(f"  Problem categories (<70%): {len(problem_categories)}")
    print(f"  Excellent categories (>90%): {len(improved_categories)}")
    
    if problem_categories:
        print(f"\nCategories needing attention:")
        for cat, acc in list(problem_categories.items())[:3]:
            print(f"    {cat}: {acc:.1%}")
    
    print("\nENHANCEMENTS IMPLEMENTED:")
    print("  1. Enhanced class balancing with SMOTE")
    print("  2. Focal Loss for imbalanced data")
    print("  3. Improved demographic inference")
    print("  4. Comprehensive bias analysis")
    print("  5. Multi-attribute debiasing")
    
    print("\nPROJECT READY FOR SUBMISSION!")
    print("=" * 60)


def enhanced_main():
    """Final enhanced main training pipeline"""
    print("=" * 60)
    print("FINAL ENHANCED RESUME CLASSIFICATION SYSTEM")
    print("=" * 60)
    print("CAI 6605: Trustworthy AI Systems - FINAL PROJECT")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    print("Target: >85% Accuracy | Enhanced RoBERTa-base")
    print("=" * 60)
    
    Config.display_enhanced_config()
    
    setup_environment()
    
    if not download_dataset():
        print("Failed to download dataset. Exiting...")
        return
    
    df, label_map, num_labels = load_and_preprocess_data(Config.DATA_PATH)
    if df is None:
        print("Failed to process data. Exiting...")
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
    print("Computed class weights for imbalanced data")
    
    training_args = TrainingArguments(
        output_dir=Config.MODEL_SAVE_PATH,
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
    
    print("\n" + "=" * 60)
    print("FINAL ENHANCED MODEL TRAINING")
    print("=" * 60)
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Effective Batch Size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
    print(f"Focal Loss: {Config.USE_FOCAL_LOSS}")
    print("=" * 60)
    
    trainer.train()
    print("\nEnhanced Model Training Complete!")
    
    trainer.save_model(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    print(f"Enhanced model saved to {Config.MODEL_SAVE_PATH}")
    
    test_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    with open('results/final_training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    with open('results/training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print_final_summary(test_results)
    
    return trainer, tokenizer, test_results


if __name__ == "__main__":
    enhanced_main()
