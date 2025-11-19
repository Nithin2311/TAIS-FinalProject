"""
Enhanced Training Script for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import os
import warnings
warnings.filterwarnings('ignore')

from enhanced_config import EnhancedConfig
from enhanced_data_processor import download_dataset, load_and_enhance_data, create_balanced_split
from enhanced_model_trainer import EnhancedResumeDataset, EnhancedCustomTrainer, enhanced_compute_metrics, enhanced_evaluate_model, setup_enhanced_model
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


def save_enhanced_training_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map):
    """Save enhanced training data for bias analysis"""
    training_data = {
        'X_train': X_train,
        'X_val': X_val, 
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_map': label_map
    }
    
    with open('data/processed/enhanced_training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    
    print("Enhanced training data saved for bias analysis")


def main():
    """Enhanced main training pipeline"""
    print("=" * 60)
    print("ENHANCED RESUME CLASSIFICATION SYSTEM TRAINING")
    print("=" * 60)
    print("CAI 6605: Trustworthy AI Systems")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    print("Target: >85% Accuracy | Model: RoBERTa-base")
    print("=" * 60)
    
    # Display enhanced configuration
    EnhancedConfig.display_config()
    
    # Setup environment
    setup_environment()
    
    # Download dataset
    if not download_dataset():
        print("Failed to download dataset. Exiting...")
        return
    
    # Load and enhance data
    df, label_map, num_labels = load_and_enhance_data(EnhancedConfig.DATA_PATH)
    if df is None:
        print("Failed to process data. Exiting...")
        return
    
    # Create balanced split
    X_train, X_val, X_test, y_train, y_val, y_test = create_balanced_split(
        df, EnhancedConfig.TEST_SIZE, EnhancedConfig.VAL_SIZE, EnhancedConfig.RANDOM_STATE
    )
    
    # Save enhanced label map and training data
    with open('data/processed/enhanced_label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    save_enhanced_training_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map)
    
    # Enhanced model setup
    model, tokenizer, device = setup_enhanced_model(num_labels, EnhancedConfig.MODEL_NAME)
    
    # Create enhanced datasets
    train_dataset = EnhancedResumeDataset(X_train, y_train, tokenizer, EnhancedConfig.MAX_LENGTH)
    val_dataset = EnhancedResumeDataset(X_val, y_val, tokenizer, EnhancedConfig.MAX_LENGTH)
    test_dataset = EnhancedResumeDataset(X_test, y_test, tokenizer, EnhancedConfig.MAX_LENGTH)
    
    # Compute class weights for imbalanced data
    if EnhancedConfig.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = class_weights.astype(np.float32)
        print("Computed class weights for imbalanced data")
    else:
        class_weights = None
        print("Using standard cross-entropy loss")
    
    # Enhanced training configuration
    training_args = TrainingArguments(
        output_dir=EnhancedConfig.MODEL_SAVE_PATH,
        num_train_epochs=EnhancedConfig.NUM_EPOCHS,
        per_device_train_batch_size=EnhancedConfig.BATCH_SIZE,
        per_device_eval_batch_size=EnhancedConfig.BATCH_SIZE * 2,
        learning_rate=EnhancedConfig.LEARNING_RATE,
        warmup_ratio=EnhancedConfig.WARMUP_RATIO,
        weight_decay=EnhancedConfig.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=EnhancedConfig.RANDOM_STATE,
        gradient_accumulation_steps=EnhancedConfig.GRADIENT_ACCUMULATION_STEPS,
        report_to="none",
        dataloader_pin_memory=False,
    )
    
    # Enhanced early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EnhancedConfig.EARLY_STOPPING_PATIENCE
    )
    
    # Create enhanced trainer
    trainer = EnhancedCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=enhanced_compute_metrics,
        class_weights=class_weights,
        callbacks=[early_stopping]
    )
    
    # Enhanced training
    print("\n" + "=" * 60)
    print("ENHANCED MODEL TRAINING STARTED")
    print("=" * 60)
    print(f"Epochs: {EnhancedConfig.NUM_EPOCHS}")
    print(f"Batch Size: {EnhancedConfig.BATCH_SIZE}")
    print(f"Learning Rate: {EnhancedConfig.LEARNING_RATE}")
    print(f"Early Stopping Patience: {EnhancedConfig.EARLY_STOPPING_PATIENCE}")
    print(f"Use Class Weights: {EnhancedConfig.USE_CLASS_WEIGHTS}")
    print("=" * 60)
    
    # Train enhanced model
    trainer.train()
    print("\nEnhanced Model Training Complete!")
    
    # Save enhanced model
    trainer.save_model(EnhancedConfig.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(EnhancedConfig.MODEL_SAVE_PATH)
    print(f"Enhanced model saved to {EnhancedConfig.MODEL_SAVE_PATH}")
    
    # Enhanced evaluation
    test_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    # Save enhanced results
    with open('results/enhanced_training_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Project summary
    print("\n" + "=" * 60)
    print("ENHANCED MODEL TRAINING COMPLETE!")
    print("=" * 60)
    accuracy = test_results['eval_accuracy']
    print(f"FINAL TEST ACCURACY: {accuracy*100:.2f}%")
    
    if accuracy > 0.85:
        print("🎯 TARGET EXCEEDED: >85% accuracy")
    elif accuracy > 0.80:
        print("✅ TARGET ACHIEVED: >80% accuracy")
    else:
        print("⚠️  TARGET NOT MET: <80% accuracy")
    
    # Show improvement opportunities
    problematic_cats = test_results.get('problematic_categories', [])
    if problematic_cats:
        print(f"\n📊 Focus Areas for Improvement:")
        for cat, acc in problematic_cats[:5]:  # Show top 5 problematic categories
            print(f"  {cat}: {acc*100:.1f}% accuracy")
    
    print("\nNext steps:")
    print("Run enhanced bias analysis: python enhanced_bias_analysis.py")
    print("Launch enhanced web interface: python enhanced_gradio_app.py")
    print("=" * 60)
    
    return trainer, tokenizer, test_results


if __name__ == "__main__":
    main()
