"""
Training script for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import os
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import download_dataset, load_and_preprocess_data, split_data
from model_trainer import ResumeDataset, CustomTrainer, compute_metrics, evaluate_model, setup_model
import torch
from transformers import TrainingArguments
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


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("RESUME CLASSIFICATION SYSTEM - MODEL TRAINING")
    print("=" * 60)
    print("CAI 6605: Trustworthy AI Systems")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    print("Target: >80% Accuracy | Model: RoBERTa-base")
    print("=" * 60)
    
    # Display configuration
    Config.display_config()
    
    # Setup environment
    setup_environment()
    
    # Download dataset
    if not download_dataset():
        print("Failed to download dataset. Exiting...")
        return
    
    # Load and preprocess data
    df, label_map, num_labels = load_and_preprocess_data(Config.DATA_PATH)
    if df is None:
        print("Failed to process data. Exiting...")
        return
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, Config.TEST_SIZE, Config.VAL_SIZE, Config.RANDOM_STATE
    )
    
    # Save label map and training data
    with open('data/processed/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    save_training_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map)
    
    # Model setup
    model, tokenizer, device = setup_model(num_labels, Config.MODEL_NAME)
    
    # Create datasets
    train_dataset = ResumeDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = class_weights.astype(np.float32)
    print("Computed class weights for imbalanced data")
    
    # Training configuration - FIXED: Updated parameter names for newer Transformers version
    training_args = TrainingArguments(
        output_dir=Config.MODEL_SAVE_PATH,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        eval_strategy="epoch",  # Fixed: changed from evaluation_strategy
        save_strategy="epoch",  # Fixed: parameter name remains the same
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=Config.RANDOM_STATE,
        report_to="none"
    )
    
    # Early stopping callback
    from transformers import EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[early_stopping]
    )
    
    # Training
    print("\n" + "=" * 60)
    print("MODEL TRAINING STARTED")
    print("=" * 60)
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
    print("=" * 60)
    
    # Train model
    trainer.train()
    print("\nTraining Complete!")
    
    # Save model
    trainer.save_model(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")
    
    # Standard evaluation
    test_results = evaluate_model(trainer, test_dataset, label_map)
    
    # Save test results
    with open('results/training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Project summary
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    accuracy = test_results['eval_accuracy']
    print(f"FINAL TEST ACCURACY: {accuracy*100:.2f}%")
    
    if accuracy > 0.80:
        print("TARGET ACHIEVED: >80% accuracy")
    else:
        print("TARGET NOT MET: <80% accuracy")
    
    print("\nNext steps:")
    print("Run bias analysis: python bias_analysis.py")
    print("Launch web interface: python gradio_app.py")
    print("=" * 60)
    
    return trainer, tokenizer, test_results


if __name__ == "__main__":
    main()
