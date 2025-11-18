"""
Enhanced Bias Analysis Script with Baseline and Debiased Model Comparison
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import json
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_trainer import ResumeDataset
from bias_analyzer import BiasAnalyzer, BiasVisualization, DemographicInference
from debiasing_experiments import PreprocessingDebiasing


def load_training_data():
    """Load training data saved during model training"""
    try:
        with open('data/processed/training_data.pkl', 'rb') as f:
            training_data = pickle.load(f)
        
        print("Training data loaded successfully")
        return training_data
    except Exception as e:
        print(f"Failed to load training data: {e}")
        print("Please run 'python train.py' first to train the model")
        return None


def load_model_and_tokenizer(model_path='models/resume_classifier'):
    """Load the trained model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"Model and tokenizer loaded successfully from {model_path}")
        return model, tokenizer, device
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None, None, None


def get_predictions(model, dataset, device):
    """Get model predictions for dataset"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs = dataset[i]
            input_ids = inputs['input_ids'].unsqueeze(0).to(device)
            attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_predictions.append(outputs.logits.cpu().numpy())
    
    return np.vstack(all_predictions)


def run_bias_analysis_for_model(model_type='baseline', training_data=None):
    """Run bias analysis for specified model type"""
    print(f"\n" + "=" * 60)
    print(f"RUNNING BIAS ANALYSIS FOR {model_type.upper()} MODEL")
    print("=" * 60)
    
    if training_data is None:
        training_data = load_training_data()
        if training_data is None:
            return None
    
    # Determine model path
    if model_type == 'baseline':
        model_path = 'models/resume_classifier'
    elif model_type == 'debiased':
        model_path = 'models/resume_classifier_debiased'
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if model is None:
        print(f"Failed to load {model_type} model. Please train the model first.")
        return None
    
    # Extract data
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    label_map = training_data['label_map']
    
    # Create test dataset and get predictions
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, 512)
    predictions = get_predictions(model, test_dataset, device)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Run bias analysis
    bias_analyzer = BiasAnalyzer(model, tokenizer, label_map, device)
    bias_report = bias_analyzer.comprehensive_bias_analysis(X_test, y_test, pred_labels)
    
    # Add accuracy to report
    accuracy = np.mean(pred_labels == y_test)
    bias_report['test_accuracy'] = accuracy
    
    # Save bias report
    report_path = f'results/{model_type}_bias_report.json'
    with open(report_path, 'w') as f:
        json.dump(bias_report, f, indent=2)
    
    # Generate visualizations
    os.makedirs(f'visualizations/{model_type}', exist_ok=True)
    try:
        BiasVisualization.plot_fairness_metrics(
            bias_report['fairness_metrics'],
            save_path=f'visualizations/{model_type}/fairness_metrics.png'
        )
        
        BiasVisualization.plot_category_bias(
            bias_report['category_bias_analysis'],
            save_path=f'visualizations/{model_type}/category_bias.png'
        )
        print(f"Visualizations saved to 'visualizations/{model_type}/' directory")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    print(f"Bias analysis complete for {model_type} model")
    return bias_report


def train_debiased_model(training_data):
    """Train debiased model using preprocessing debiasing"""
    print("\n" + "=" * 60)
    print("TRAINING DEBIASED MODEL")
    print("=" * 60)
    
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    X_val = training_data['X_val']
    y_val = training_data['y_val']
    label_map = training_data['label_map']
    
    # Infer demographics for training data
    print("Inferring demographics for debiasing...")
    demo_inference = DemographicInference()
    train_demographics = {
        'gender': [demo_inference.infer_gender(text) for text in X_train]
    }
    
    # Apply preprocessing debiasing
    preprocessor = PreprocessingDebiasing()
    
    print("Applying preprocessing debiasing...")
    debiased_texts = [preprocessor.remove_demographic_indicators(text) for text in X_train]
    
    # Balance dataset
    balanced_texts, balanced_labels = preprocessor.balance_dataset(
        debiased_texts, y_train, train_demographics, 'gender'
    )
    
    print(f"Debiased dataset: {len(balanced_texts)} samples (was {len(X_train)})")
    
    # Train new model on debiased data
    from train import setup_model, ResumeDataset, CustomTrainer, compute_metrics, evaluate_model
    from transformers import TrainingArguments, EarlyStoppingCallback
    from sklearn.utils.class_weight import compute_class_weight
    
    num_labels = len(label_map)
    model, tokenizer, device = setup_model(num_labels, 'roberta-base')
    
    # Create datasets with debiased data
    train_dataset = ResumeDataset(balanced_texts, balanced_labels, tokenizer, 512)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, 512)
    
    # Compute class weights for balanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(balanced_labels),
        y=balanced_labels
    )
    class_weights = class_weights.astype(np.float32)
    
    # Training arguments for debiased model
    training_args = TrainingArguments(
        output_dir='models/resume_classifier_debiased',
        num_train_epochs=20,  # Reduced for faster training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=42,
        report_to="none"
    )
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    
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
    
    # Train debiased model
    print("Training debiased model...")
    trainer.train()
    
    # Save debiased model
    trainer.save_model('models/resume_classifier_debiased')
    tokenizer.save_pretrained('models/resume_classifier_debiased')
    
    # Evaluate debiased model
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, 512)
    debiased_results = evaluate_model(trainer, test_dataset, label_map)
    
    # Save results
    with open('results/debiased_results.json', 'w') as f:
        json.dump(debiased_results, f, indent=2)
    
    print(f"Debiased model trained with accuracy: {debiased_results['eval_accuracy']:.4f}")
    
    return {
        'trainer': trainer,
        'tokenizer': tokenizer,
        'results': debiased_results
    }


def compare_models():
    """Compare baseline and debiased models"""
    print("\n" + "=" * 60)
    print("COMPARING BASELINE vs DEBIASED MODELS")
    print("=" * 60)
    
    # Load results
    try:
        with open('results/baseline_bias_report.json', 'r') as f:
            baseline_report = json.load(f)
        
        with open('results/debiased_bias_report.json', 'r') as f:
            debiased_report = json.load(f)
        
        with open('results/training_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        with open('results/debiased_results.json', 'r') as f:
            debiased_results = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None
    
    # Calculate comparison metrics
    comparison = {
        'performance': {
            'baseline_accuracy': baseline_results['eval_accuracy'],
            'debiased_accuracy': debiased_results['eval_accuracy'],
            'accuracy_change': debiased_results['eval_accuracy'] - baseline_results['eval_accuracy'],
            'accuracy_change_percent': ((debiased_results['eval_accuracy'] - baseline_results['eval_accuracy']) / baseline_results['eval_accuracy']) * 100
        },
        'fairness_improvement': {},
        'bias_reduction': {}
    }
    
    # Fairness metrics comparison
    for demo_type in baseline_report['fairness_metrics'].keys():
        if demo_type in debiased_report['fairness_metrics']:
            baseline_fairness = np.mean(list(baseline_report['fairness_metrics'][demo_type].values()))
            debiased_fairness = np.mean(list(debiased_report['fairness_metrics'][demo_type].values()))
            comparison['fairness_improvement'][demo_type] = {
                'baseline': baseline_fairness,
                'debiased': debiased_fairness,
                'improvement': debiased_fairness - baseline_fairness
            }
    
    # Bias reduction
    baseline_name_bias = baseline_report['name_substitution_bias']['average_gender_bias']
    debiased_name_bias = debiased_report['name_substitution_bias']['average_gender_bias']
    
    comparison['bias_reduction'] = {
        'name_bias': {
            'baseline': baseline_name_bias,
            'debiased': debiased_name_bias,
            'reduction': baseline_name_bias - debiased_name_bias,
            'reduction_percent': ((baseline_name_bias - debiased_name_bias) / baseline_name_bias) * 100 if baseline_name_bias > 0 else 0
        }
    }
    
    # Overall assessment
    accuracy_improvement = max(0, comparison['performance']['accuracy_change_percent']) / 100
    fairness_improvement = np.mean([v['improvement'] for v in comparison['fairness_improvement'].values()]) if comparison['fairness_improvement'] else 0
    bias_reduction = comparison['bias_reduction']['name_bias']['reduction_percent'] / 100
    
    overall_score = (accuracy_improvement * 0.4 + fairness_improvement * 0.4 + bias_reduction * 0.2)
    
    if overall_score > 0.3:
        rating = "EXCELLENT"
    elif overall_score > 0.1:
        rating = "GOOD"
    elif overall_score > 0:
        rating = "MODEST"
    else:
        rating = "NEGATIVE"
    
    comparison['overall_assessment'] = {
        'score': overall_score,
        'rating': rating,
        'summary': f"Debiasing effectiveness: {rating}"
    }
    
    # Save comparison
    with open('results/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print_comparison_summary(comparison)
    
    # Generate comparison visualizations
    generate_comparison_visualizations(baseline_report, debiased_report, comparison)
    
    return comparison


def print_comparison_summary(comparison):
    """Print comprehensive comparison summary"""
    print("\nCOMPARISON RESULTS:")
    print("=" * 50)
    
    perf = comparison['performance']
    print(f"\nPERFORMANCE:")
    print(f"  Baseline Accuracy: {perf['baseline_accuracy']:.3f}")
    print(f"  Debiased Accuracy: {perf['debiased_accuracy']:.3f}")
    print(f"  Change: {perf['accuracy_change_percent']:+.2f}%")
    
    bias_red = comparison['bias_reduction']['name_bias']
    print(f"\nBIAS REDUCTION:")
    print(f"  Baseline Name Bias: {bias_red['baseline']:.3f}")
    print(f"  Debiased Name Bias: {bias_red['debiased']:.3f}")
    print(f"  Reduction: {bias_red['reduction_percent']:+.2f}%")
    
    print(f"\nFAIRNESS IMPROVEMENTS:")
    for demo_type, improvement in comparison['fairness_improvement'].items():
        arrow = "↗️" if improvement['improvement'] > 0 else "↘️"
        print(f"  {demo_type.upper():20s}: {improvement['improvement']:+.3f} {arrow}")
    
    assessment = comparison['overall_assessment']
    print(f"\nOVERALL ASSESSMENT: {assessment['rating']} (Score: {assessment['score']:.3f})")
    print("=" * 50)


def generate_comparison_visualizations(baseline_report, debiased_report, comparison_results):
    """Generate visualizations comparing before and after debiasing"""
    try:
        import matplotlib.pyplot as plt
        
        # Fairness metrics comparison
        demo_types = list(baseline_report['fairness_metrics'].keys())
        metrics = ['demographic_parity', 'equal_opportunity', 'accuracy_equality']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            baseline_values = [baseline_report['fairness_metrics'][demo][metric] for demo in demo_types]
            debiased_values = [debiased_report['fairness_metrics'][demo][metric] for demo in demo_types]
            
            x = np.arange(len(demo_types))
            width = 0.35
            
            axes[idx].bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7, color='red')
            axes[idx].bar(x + width/2, debiased_values, width, label='Debiased', alpha=0.7, color='green')
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[idx].set_ylabel('Disparity Score')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(demo_types, rotation=45)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/fairness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance comparison
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'Name Bias']
        baseline_metrics = [
            baseline_report['test_accuracy'],
            baseline_report['name_substitution_bias']['average_gender_bias']
        ]
        debiased_metrics = [
            debiased_report['test_accuracy'],
            debiased_report['name_substitution_bias']['average_gender_bias']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, baseline_metrics, width, label='Baseline', alpha=0.7, color='red')
        plt.bar(x + width/2, debiased_metrics, width, label='Debiased', alpha=0.7, color='green')
        plt.title('Performance and Bias Comparison')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/performance_bias_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comparison visualizations saved to 'visualizations/' directory")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")


def run_comprehensive_bias_analysis_with_comparison():
    """Run comprehensive bias analysis with debiasing comparison"""
    print("=" * 60)
    print("COMPREHENSIVE BIAS ANALYSIS WITH MODEL COMPARISON")
    print("=" * 60)
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    print("=" * 60)
    
    # Load training data
    training_data = load_training_data()
    if training_data is None:
        return None
    
    # Step 1: Run bias analysis for baseline model
    baseline_report = run_bias_analysis_for_model('baseline', training_data)
    if baseline_report is None:
        print("Failed to analyze baseline model. Please ensure baseline model is trained.")
        return None
    
    # Step 2: Train debiased model (if not exists)
    if not os.path.exists('models/resume_classifier_debiased'):
        print("\nDebiased model not found. Training debiased model...")
        debiased_result = train_debiased_model(training_data)
        if debiased_result is None:
            print("Failed to train debiased model.")
            return None
    else:
        print("\nDebiased model found. Skipping training...")
    
    # Step 3: Run bias analysis for debiased model
    debiased_report = run_bias_analysis_for_model('debiased', training_data)
    if debiased_report is None:
        print("Failed to analyze debiased model.")
        return None
    
    # Step 4: Compare models
    comparison = compare_models()
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'baseline_report': baseline_report,
        'debiased_report': debiased_report,
        'comparison': comparison
    }


def main():
    """Enhanced main function with dual model analysis"""
    # Run comprehensive analysis with comparison
    results = run_comprehensive_bias_analysis_with_comparison()
    
    if results is None:
        return
    
    print("\n" + "=" * 60)
    print("FINAL PROJECT ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Final recommendations
    if 'comparison' in results:
        rating = results['comparison']['overall_assessment']['rating']
        print(f"\nFINAL RATING: {rating}")
        
        if rating in ['EXCELLENT', 'GOOD']:
            print("✅ Debiasing strategies were effective")
            print("✅ Model maintains good performance while reducing bias")
        else:
            print("⚠️  Debiasing had limited effectiveness")
            print("⚠️  Consider additional strategies or data collection")
    
    print("\nKey Deliverables:")
    print("  ✅ Baseline model bias analysis")
    print("  ✅ Debiased model training and analysis") 
    print("  ✅ Comprehensive before/after comparison")
    print("  ✅ Professional visualizations and reporting")
    
    print("\nFiles Generated:")
    print("  results/baseline_bias_report.json")
    print("  results/debiased_bias_report.json") 
    print("  results/model_comparison.json")
    print("  visualizations/baseline/ - Baseline charts")
    print("  visualizations/debiased/ - Debiased charts")
    print("  visualizations/fairness_comparison.png")
    print("  visualizations/performance_bias_comparison.png")
    
    print("\nNext Steps:")
    print("  Launch enhanced Gradio interface: python gradio_app.py")
    print("  Review the comprehensive comparison report")
    print("  =============================================")


if __name__ == "__main__":
    main()
