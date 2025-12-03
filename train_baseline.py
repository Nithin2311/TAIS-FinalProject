"""
Baseline model training script.
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
from model_trainer import ResumeDataset, CustomTrainer, compute_metrics, evaluate_model, setup_model


def setup_environment():
    """Create necessary directories"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)


def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
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
    
    print("Training data saved")
    
    summary = {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'num_classes': len(label_map),
        'class_distribution': {
            'train': dict(zip([str(k) for k in np.unique(y_train)], np.unique(y_train, return_counts=True)[1].tolist())),
            'val': dict(zip([str(k) for k in np.unique(y_val)], np.unique(y_val, return_counts=True)[1].tolist())),
            'test': dict(zip([str(k) for k in np.unique(y_test)], np.unique(y_test, return_counts=True)[1].tolist()))
        }
    }
    
    with open('data/processed/training_data_summary.json', 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)


def train_baseline():
    """Train baseline model"""
    print("Baseline Resume Classification System Training")
    
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
    
    label_map_str = {str(k): v for k, v in label_map.items()}
    
    with open('data/processed/label_map.json', 'w') as f:
        json.dump(label_map_str, f, indent=2)
    
    save_training_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map_str)
    
    model, tokenizer, device = setup_model(num_labels, Config.MODEL_NAME)
    
    train_dataset = ResumeDataset(X_train, y_train, tokenizer, Config.MAX_LENGTH)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, Config.MAX_LENGTH)
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, Config.MAX_LENGTH)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = class_weights.astype(np.float32)
    
    training_args = TrainingArguments(
        output_dir=Config.BASELINE_MODEL_PATH,
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
        save_total_limit=2,
        label_names=["labels"]
    )
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=Config.USE_FOCAL_LOSS,
        callbacks=[early_stopping]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    trainer.train()
    print("Training complete")
    
    trainer.save_model(Config.BASELINE_MODEL_PATH)
    tokenizer.save_pretrained(Config.BASELINE_MODEL_PATH)
    print(f"Model saved to {Config.BASELINE_MODEL_PATH}")
    
    test_results = evaluate_model(trainer, test_dataset, label_map_str)
    
    test_results_serializable = convert_numpy_types(test_results)
    
    with open('results/baseline_results.json', 'w') as f:
        json.dump(test_results_serializable, f, indent=2)
    
    accuracy = test_results['eval_accuracy']
    f1 = test_results['eval_f1']
    macro_f1 = test_results['eval_macro_f1']
    
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    category_accuracies = test_results['per_class_accuracy']
    problem_categories = {cat: acc for cat, acc in category_accuracies.items() if acc < 0.7}
    
    if problem_categories:
        print(f"\nCategories with accuracy < 70%:")
        for cat, acc in sorted(problem_categories.items(), key=lambda x: x[1]):
            print(f"  {cat}: {acc*100:.1f}%")
    
    stats = {
        'model_name': Config.MODEL_NAME,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'config': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'epochs': Config.NUM_EPOCHS,
            'dropout': Config.DROPOUT_RATE
        }
    }
    
    with open('results/model_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return trainer, tokenizer, test_results_serializable


if __name__ == "__main__":
    train_baseline()
