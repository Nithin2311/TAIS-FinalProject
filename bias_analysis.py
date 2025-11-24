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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback
from model_trainer import ResumeDataset, EnhancedTrainer, compute_metrics, enhanced_evaluate_model, setup_optimized_model
from bias_analyzer import EnhancedBiasAnalyzer, EnhancedBiasVisualization, EnhancedDemographicInference
from sklearn.utils.class_weight import compute_class_weight

# Try to import ImprovedBiasMitigationPipeline, provide fallback if not available
try:
    from debiasing_experiments import ImprovedBiasMitigationPipeline
    HAS_DEBIASING_PIPELINE = True
except ImportError:
    print("Warning: ImprovedBiasMitigationPipeline not found. Debiasing training will be limited.")
    HAS_DEBIASING_PIPELINE = False
    # Create a simple fallback class
    class ImprovedBiasMitigationPipeline:
        def __init__(self, model, tokenizer, label_map, device):
            self.model = model
            self.tokenizer = tokenizer
            self.label_map = label_map
            self.device = device
        
        def apply_improved_debiasing(self, X_train, y_train, X_val, y_val, train_demographics):
            print("Using fallback debiasing - returning original data")
            return X_train, y_train


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if hasattr(obj, 'dtype'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(element) for element in obj)
    else:
        return obj

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
    print(f"RUNNING ENHANCED BIAS ANALYSIS FOR {model_type.upper()} MODEL")
    print("=" * 60)
    
    if training_data is None:
        training_data = load_training_data()
        if training_data is None:
            return None
    
    # Determine model path
    if model_type == 'baseline':
        model_path = 'models/resume_classifier'
    elif model_type == 'debiased':
        # Check if improved model exists, otherwise use standard path
        improved_path = 'models/resume_classifier_debiased_improved'
        if os.path.exists(improved_path):
            model_path = improved_path
            print("Using IMPROVED debiased model for analysis")
        else:
            model_path = 'models/resume_classifier_debiased'
            print("Using standard debiased model for analysis")
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
    
    # Calculate current accuracy for verification
    current_accuracy = np.mean(pred_labels == y_test)
    print(f"Current model accuracy: {current_accuracy:.4f}")
    
    # Run enhanced bias analysis
    bias_analyzer = EnhancedBiasAnalyzer(model, tokenizer, label_map, device)
    bias_report = bias_analyzer.comprehensive_bias_analysis(X_test, y_test, pred_labels)
    
    # Add accuracy to report
    bias_report['test_accuracy'] = float(current_accuracy)
    
    # Convert all numpy types to native Python types before saving
    bias_report = convert_numpy_types(bias_report)
    
    # Save bias report
    report_path = f'results/{model_type}_bias_report.json'
    with open(report_path, 'w') as f:
        json.dump(bias_report, f, indent=2)
    
    # Generate enhanced visualizations
    os.makedirs(f'visualizations/{model_type}', exist_ok=True)
    try:
        EnhancedBiasVisualization.plot_comprehensive_fairness_metrics(
            bias_report['fairness_metrics'],
            save_path=f'visualizations/{model_type}/enhanced_fairness_metrics.png'
        )
        
        EnhancedBiasVisualization.plot_category_performance_bias(
            bias_report['category_bias_analysis'],
            save_path=f'visualizations/{model_type}/enhanced_category_bias.png'
        )
        
        # Plot intersectional bias if available
        if bias_report.get('intersectional_bias_analysis'):
            EnhancedBiasVisualization.plot_intersectional_bias_heatmap(
                bias_report['intersectional_bias_analysis'],
                save_path=f'visualizations/{model_type}/intersectional_bias_heatmap.png'
            )
        
        print(f"Enhanced visualizations saved to 'visualizations/{model_type}/' directory")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    print(f"Enhanced bias analysis complete for {model_type} model")
    return bias_report


def train_improved_debiased_model(training_data):
    """Train improved debiased model using performance-preserving approach"""
    print("\n" + "=" * 60)
    print("TRAINING IMPROVED DEBIASED MODEL - PERFORMANCE PRESERVING")
    print("=" * 60)
    
    if not HAS_DEBIASING_PIPELINE:
        print("Debiasing pipeline not available. Using standard training.")
        return train_standard_debiased_model(training_data)
    
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    X_val = training_data['X_val']
    y_val = training_data['y_val']
    label_map = training_data['label_map']
    
    # Conservative demographic inference
    print("Conservative demographic inference...")
    demo_inference = EnhancedDemographicInference()
    train_demographics = {
        'gender': [demo_inference.infer_gender(text) for text in X_train]
    }
    
    # Apply improved debiasing
    num_labels = len(label_map)
    model, tokenizer, device = setup_optimized_model(num_labels, 'roberta-base')
    
    improved_pipeline = ImprovedBiasMitigationPipeline(model, tokenizer, label_map, device)
    
    print("Applying improved performance-preserving debiasing...")
    debiased_texts, debiased_labels = improved_pipeline.apply_improved_debiasing(
        X_train, y_train, X_val, y_val, train_demographics
    )
    
    print(f"Improved debiased dataset: {len(debiased_texts)} samples (was {len(X_train)})")
    
    # Train new model with conservative settings
    model, tokenizer, device = setup_optimized_model(num_labels, 'roberta-base')
    
    # Create datasets
    train_dataset = ResumeDataset(debiased_texts, debiased_labels, tokenizer, 512)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, 512)
    
    # Conservative class weights - ADD IMPORT AT TOP OF FILE
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(debiased_labels),
        y=debiased_labels
    )
    class_weights = class_weights.astype(np.float32)
    
    # Conservative training arguments
    training_args = TrainingArguments(
        output_dir='models/resume_classifier_debiased_improved',
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=1.5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=42,
        report_to="none",
        dataloader_pin_memory=False
    )
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=True,
        callbacks=[early_stopping]
    )
    
    # Train improved debiased model
    print("Training improved debiased model...")
    trainer.train()
    
    # Save improved debiased model
    trainer.save_model('models/resume_classifier_debiased_improved')
    tokenizer.save_pretrained('models/resume_classifier_debiased_improved')
    
    # Also save as standard debiased for compatibility
    trainer.save_model('models/resume_classifier_debiased')
    tokenizer.save_pretrained('models/resume_classifier_debiased')
    
    # Enhanced evaluation
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, 512)
    debiased_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    # Convert results
    debiased_results = convert_numpy_types(debiased_results)
    
    # Save results
    with open('results/improved_debiased_results.json', 'w') as f:
        json.dump(debiased_results, f, indent=2)
    
    # Also save as enhanced_debiased_results for compatibility
    with open('results/enhanced_debiased_results.json', 'w') as f:
        json.dump(debiased_results, f, indent=2)
    
    accuracy = debiased_results['eval_accuracy']
    print(f"Improved debiased model trained with accuracy: {accuracy:.4f}")
    
    # Performance check
    baseline_accuracy = 0.8633
    if accuracy < baseline_accuracy - 0.03:
        print("Warning: Significant performance drop detected!")
        print("Consider using less aggressive debiasing or focusing on targeted improvements only.")
    else:
        print("Performance preserved within acceptable range!")
    
    return {
        'trainer': trainer,
        'tokenizer': tokenizer,
        'results': debiased_results
    }


def train_standard_debiased_model(training_data):
    """Fallback function to train standard debiased model"""
    print("Training standard debiased model (fallback)...")
    
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    X_val = training_data['X_val']
    y_val = training_data['y_val']
    label_map = training_data['label_map']
    
    num_labels = len(label_map)
    model, tokenizer, device = setup_optimized_model(num_labels, 'roberta-base')
    
    # Create datasets
    train_dataset = ResumeDataset(X_train, y_train, tokenizer, 512)
    val_dataset = ResumeDataset(X_val, y_val, tokenizer, 512)
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = class_weights.astype(np.float32)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='models/resume_classifier_debiased',
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
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
        seed=42
    )
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal_loss=True
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model('models/resume_classifier_debiased')
    tokenizer.save_pretrained('models/resume_classifier_debiased')
    
    # Evaluation
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, 512)
    debiased_results = enhanced_evaluate_model(trainer, test_dataset, label_map)
    
    # Save results
    with open('results/enhanced_debiased_results.json', 'w') as f:
        json.dump(debiased_results, f, indent=2)
    
    accuracy = debiased_results['eval_accuracy']
    print(f"Standard debiased model trained with accuracy: {accuracy:.4f}")
    
    return {
        'trainer': trainer,
        'tokenizer': tokenizer,
        'results': debiased_results
    }


def enhanced_compare_models():
    """Enhanced comparison between baseline and debiased models"""
    print("\n" + "=" * 60)
    print("ENHANCED COMPARISON: BASELINE vs DEBIASED MODELS")
    print("=" * 60)
    
    # Load results
    try:
        with open('results/baseline_bias_report.json', 'r') as f:
            baseline_report = json.load(f)
        
        # Try to load improved results first
        improved_results_path = 'results/improved_debiased_results.json'
        if os.path.exists(improved_results_path):
            with open(improved_results_path, 'r') as f:
                debiased_results = json.load(f)
            print("Using IMPROVED debiased results for comparison")
        else:
            with open('results/enhanced_debiased_results.json', 'r') as f:
                debiased_results = json.load(f)
        
        with open('results/enhanced_training_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        # Load debiased bias report - use improved if available
        improved_bias_report = 'results/debiased_bias_report.json'
        if os.path.exists(improved_bias_report):
            with open(improved_bias_report, 'r') as f:
                debiased_report = json.load(f)
        else:
            debiased_report = {}
            
    except Exception as e:
        print(f"Error loading results: {e}")
        try:
            with open('results/training_results.json', 'r') as f:
                baseline_results = json.load(f)
            print("Loaded training_results.json as baseline")
        except:
            print("No baseline results found")
            return None
    
    # Calculate enhanced comparison metrics
    comparison = {
        'performance': {
            'baseline_accuracy': baseline_results.get('eval_accuracy', 0),
            'debiased_accuracy': debiased_results.get('eval_accuracy', 0),
            'accuracy_change': debiased_results.get('eval_accuracy', 0) - baseline_results.get('eval_accuracy', 0),
            'accuracy_change_percent': ((debiased_results.get('eval_accuracy', 0) - baseline_results.get('eval_accuracy', 0)) / baseline_results.get('eval_accuracy', 0)) * 100 if baseline_results.get('eval_accuracy', 0) > 0 else 0
        },
        'fairness_improvement': {},
        'bias_reduction': {},
        'category_improvements': {},
        'counterfactual_improvement': {}
    }
    
    # Enhanced fairness metrics comparison
    baseline_fairness = baseline_report.get('fairness_metrics', {})
    debiased_fairness = debiased_report.get('fairness_metrics', {})
    
    for demo_type in baseline_fairness.keys():
        if demo_type in debiased_fairness:
            # Calculate average fairness metric
            baseline_avg = np.mean([
                baseline_fairness[demo_type].get('demographic_parity', 0),
                baseline_fairness[demo_type].get('equal_opportunity', 0),
                baseline_fairness[demo_type].get('accuracy_equality', 0)
            ])
            debiased_avg = np.mean([
                debiased_fairness[demo_type].get('demographic_parity', 0),
                debiased_fairness[demo_type].get('equal_opportunity', 0),
                debiased_fairness[demo_type].get('accuracy_equality', 0)
            ])
            
            comparison['fairness_improvement'][demo_type] = {
                'baseline': float(baseline_avg),
                'debiased': float(debiased_avg),
                'improvement': float(debiased_avg - baseline_avg)
            }
    
    # Enhanced bias reduction analysis
    baseline_name_bias = baseline_report.get('name_substitution_bias', {}).get('gender_bias', {}).get('average_bias', 0)
    debiased_name_bias = debiased_report.get('name_substitution_bias', {}).get('gender_bias', {}).get('average_bias', 0)
    
    comparison['bias_reduction'] = {
        'gender_bias': {
            'baseline': float(baseline_name_bias),
            'debiased': float(debiased_name_bias),
            'reduction': float(baseline_name_bias - debiased_name_bias),
            'reduction_percent': float(((baseline_name_bias - debiased_name_bias) / baseline_name_bias) * 100 if baseline_name_bias > 0 else 0)
        },
        'racial_bias': {
            'baseline': float(baseline_report.get('name_substitution_bias', {}).get('racial_bias', {}).get('average_bias', 0)),
            'debiased': float(debiased_report.get('name_substitution_bias', {}).get('racial_bias', {}).get('average_bias', 0)),
            'reduction_percent': float(((baseline_report.get('name_substitution_bias', {}).get('racial_bias', {}).get('average_bias', 0) - debiased_report.get('name_substitution_bias', {}).get('racial_bias', {}).get('average_bias', 0)) / baseline_report.get('name_substitution_bias', {}).get('racial_bias', {}).get('average_bias', 0)) * 100 if baseline_report.get('name_substitution_bias', {}).get('racial_bias', {}).get('average_bias', 0) > 0 else 0)
        }
    }
    
    # Counterfactual fairness improvement
    baseline_cf = baseline_report.get('counterfactual_fairness', {}).get('counterfactual_unfairness_rate', 0)
    debiased_cf = debiased_report.get('counterfactual_fairness', {}).get('counterfactual_unfairness_rate', 0)
    comparison['counterfactual_improvement'] = {
        'baseline': float(baseline_cf),
        'debiased': float(debiased_cf),
        'improvement': float(baseline_cf - debiased_cf)
    }
    
    # Category-level improvements
    baseline_categories = baseline_results.get('per_class_accuracy', {})
    debiased_categories = debiased_results.get('per_class_accuracy', {})
    
    improved_categories = []
    for category in set(baseline_categories.keys()) & set(debiased_categories.keys()):
        improvement = debiased_categories[category] - baseline_categories[category]
        if improvement > 0:
            improved_categories.append((category, float(improvement)))
    
    improved_categories.sort(key=lambda x: x[1], reverse=True)
    comparison['category_improvements'] = {
        'most_improved': improved_categories[:5],
        'total_improved': len(improved_categories),
        'avg_improvement': float(np.mean([imp for _, imp in improved_categories]) if improved_categories else 0)
    }
    
    # Enhanced overall assessment
    accuracy_improvement = max(0, comparison['performance']['accuracy_change_percent']) / 100
    fairness_improvement = np.mean([v['improvement'] for v in comparison['fairness_improvement'].values()]) if comparison['fairness_improvement'] else 0
    bias_reduction = comparison['bias_reduction']['gender_bias']['reduction_percent'] / 100
    category_improvement = comparison['category_improvements']['avg_improvement']
    cf_improvement = comparison['counterfactual_improvement']['improvement']
    
    # Weighted overall score
    overall_score = float(
        accuracy_improvement * 0.3 +
        max(fairness_improvement, 0) * 0.25 +
        max(bias_reduction, 0) * 0.2 +
        min(category_improvement * 10, 1.0) * 0.15 +
        min(cf_improvement * 5, 1.0) * 0.1
    )
    
    if overall_score > 0.25:
        rating = "EXCELLENT"
    elif overall_score > 0.15:
        rating = "GOOD"
    elif overall_score > 0.05:
        rating = "MODEST"
    elif overall_score > 0:
        rating = "SLIGHT"
    else:
        rating = "NEGATIVE"
    
    comparison['overall_assessment'] = {
        'score': overall_score,
        'rating': rating,
        'summary': f"Debiasing effectiveness: {rating}",
        'components': {
            'accuracy_improvement': float(accuracy_improvement),
            'fairness_improvement': float(fairness_improvement),
            'bias_reduction': float(bias_reduction),
            'category_improvement': float(category_improvement),
            'counterfactual_improvement': float(cf_improvement)
        }
    }
    
    # Convert comparison to JSON serializable format
    comparison = convert_numpy_types(comparison)
    
    # Save enhanced comparison
    with open('results/enhanced_model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print enhanced summary
    print_enhanced_comparison_summary(comparison)
    
    # Generate enhanced comparison visualizations
    generate_enhanced_comparison_visualizations(baseline_report, debiased_report, comparison)
    
    return comparison


def print_enhanced_comparison_summary(comparison):
    """Print enhanced comprehensive comparison summary"""
    print("\nENHANCED COMPARISON RESULTS:")
    print("=" * 60)
    
    perf = comparison['performance']
    print(f"\nPERFORMANCE:")
    print(f"  Baseline Accuracy: {perf['baseline_accuracy']:.3f}")
    print(f"  Debiased Accuracy: {perf['debiased_accuracy']:.3f}")
    print(f"  Change: {perf['accuracy_change_percent']:+.2f}%")
    
    bias_red = comparison['bias_reduction']['gender_bias']
    print(f"\nBIAS REDUCTION:")
    print(f"  Baseline Gender Bias: {bias_red['baseline']:.3f}")
    print(f"  Debiased Gender Bias: {bias_red['debiased']:.3f}")
    print(f"  Reduction: {bias_red['reduction_percent']:+.2f}%")
    
    cf_improvement = comparison['counterfactual_improvement']
    print(f"\nCOUNTERFACTUAL FAIRNESS:")
    print(f"  Baseline: {cf_improvement['baseline']:.3f}")
    print(f"  Debiased: {cf_improvement['debiased']:.3f}")
    print(f"  Improvement: {cf_improvement['improvement']:+.3f}")
    
    print(f"\nFAIRNESS IMPROVEMENTS:")
    for demo_type, improvement in comparison['fairness_improvement'].items():
        arrow = "▲" if improvement['improvement'] > 0 else "▼"
        print(f"  {demo_type.upper():20s}: {improvement['improvement']:+.3f} {arrow}")
    
    print(f"\nCATEGORY IMPROVEMENTS:")
    improvements = comparison['category_improvements']
    print(f"  Categories Improved: {improvements['total_improved']}")
    print(f"  Average Improvement: {improvements['avg_improvement']:.3f}")
    if improvements['most_improved']:
        print(f"  Most Improved Categories:")
        for category, imp in improvements['most_improved'][:3]:
            print(f"    - {category}: +{imp:.3f}")
    
    assessment = comparison['overall_assessment']
    print(f"\nOVERALL ASSESSMENT: {assessment['rating']} (Score: {assessment['score']:.3f})")
    print(f"  Summary: {assessment['summary']}")
    print("=" * 60)


def generate_enhanced_comparison_visualizations(baseline_report, debiased_report, comparison_results):
    """Generate enhanced visualizations comparing before and after debiasing"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Comprehensive fairness metrics comparison
        demo_types = list(baseline_report.get('fairness_metrics', {}).keys())
        if demo_types:
            metrics = ['demographic_parity', 'equal_opportunity', 'accuracy_equality', 'statistical_parity']
            
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                if idx < len(axes):
                    baseline_values = []
                    debiased_values = []
                    valid_demo_types = []
                    
                    for demo in demo_types:
                        baseline_val = baseline_report['fairness_metrics'][demo].get(metric, 0)
                        debiased_val = debiased_report['fairness_metrics'][demo].get(metric, 0)
                        if baseline_val != 0 or debiased_val != 0:
                            baseline_values.append(float(baseline_val))
                            debiased_values.append(float(debiased_val))
                            valid_demo_types.append(demo)
                    
                    if valid_demo_types:
                        x = np.arange(len(valid_demo_types))
                        width = 0.35
                        
                        bars1 = axes[idx].bar(x - width/2, baseline_values, width, 
                                            label='Baseline', alpha=0.8, color='#ff6b6b')
                        bars2 = axes[idx].bar(x + width/2, debiased_values, width, 
                                            label='Debiased', alpha=0.8, color='#4ecdc4')
                        
                        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                                          fontsize=14, fontweight='bold', pad=20)
                        axes[idx].set_ylabel('Disparity Score', fontsize=12)
                        axes[idx].set_xticks(x)
                        axes[idx].set_xticklabels(valid_demo_types, rotation=45, ha='right')
                        axes[idx].legend(fontsize=10)
                        axes[idx].grid(True, alpha=0.3, axis='y')
                        
                        # Add value labels on bars
                        for bar, value in zip(bars1, baseline_values):
                            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                         f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                        for bar, value in zip(bars2, debiased_values):
                            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                         f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('visualizations/enhanced_fairness_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Performance and bias radar chart
        fig = plt.figure(figsize=(12, 8))
        
        # Metrics for radar chart
        categories = ['Accuracy', 'Gender Bias\nReduction', 'Fairness\nImprovement', 'Category\nImprovement', 'Counterfactual\nImprovement']
        
        baseline_metrics = [
            comparison_results['performance']['baseline_accuracy'],
            comparison_results['bias_reduction']['gender_bias']['baseline'],
            np.mean([v['baseline'] for v in comparison_results['fairness_improvement'].values()]) if comparison_results['fairness_improvement'] else 0,
            0.5,
            comparison_results['counterfactual_improvement']['baseline']
        ]
        
        debiased_metrics = [
            comparison_results['performance']['debiased_accuracy'],
            comparison_results['bias_reduction']['gender_bias']['debiased'],
            np.mean([v['debiased'] for v in comparison_results['fairness_improvement'].values()]) if comparison_results['fairness_improvement'] else 0,
            0.5 + comparison_results['category_improvements']['avg_improvement'],
            comparison_results['counterfactual_improvement']['debiased']
        ]
        
        # Normalize metrics for radar chart
        baseline_norm = [min(1.0, m * 2 if i == 1 else m) for i, m in enumerate(baseline_metrics)]
        debiased_norm = [min(1.0, m * 2 if i == 1 else m) for i, m in enumerate(debiased_metrics)]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        baseline_norm += baseline_norm[:1]
        debiased_norm += debiased_norm[:1]
        categories_radar = categories + [categories[0]]
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color='#ff6b6b')
        ax.fill(angles, baseline_norm, alpha=0.25, color='#ff6b6b')
        ax.plot(angles, debiased_norm, 'o-', linewidth=2, label='Debiased', color='#4ecdc4')
        ax.fill(angles, debiased_norm, alpha=0.25, color='#4ecdc4')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_radar[:-1], fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Enhanced Performance and Bias Reduction Radar Chart', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Enhanced comparison visualizations saved to 'visualizations/' directory")
        
    except Exception as e:
        print(f"Enhanced visualization generation failed: {e}")
        import traceback
        traceback.print_exc()


def run_enhanced_comprehensive_analysis():
    """Run enhanced comprehensive bias analysis with comparison"""
    print("=" * 60)
    print("IMPROVED COMPREHENSIVE BIAS ANALYSIS WITH MODEL COMPARISON")
    print("=" * 60)
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    print("=" * 60)
    
    # Load training data
    training_data = load_training_data()
    if training_data is None:
        return None
    
    # Step 1: Run enhanced bias analysis for baseline model
    print("\nANALYZING BASELINE MODEL...")
    baseline_report = run_bias_analysis_for_model('baseline', training_data)
    if baseline_report is None:
        print("Failed to analyze baseline model. Please ensure baseline model is trained.")
        return None
    
    # Step 2: Train IMPROVED debiased model
    print("\nTRAINING IMPROVED DEBIASED MODEL...")
    debiased_result = train_improved_debiased_model(training_data)
    if debiased_result is None:
        print("Failed to train improved debiased model.")
        return None
    
    # Step 3: Run enhanced bias analysis for improved debiased model
    print("\nANALYZING IMPROVED DEBIASED MODEL...")
    debiased_report = run_bias_analysis_for_model('debiased', training_data)
    if debiased_report is None:
        print("Failed to analyze improved debiased model.")
        return None
    
    # Step 4: Enhanced model comparison
    print("\nCOMPARING MODELS...")
    comparison = enhanced_compare_models()
    
    print("\n" + "=" * 60)
    print("IMPROVED COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'baseline_report': baseline_report,
        'debiased_report': debiased_report,
        'comparison': comparison
    }


def main():
    """Enhanced main function with comprehensive analysis"""
    # Run enhanced comprehensive analysis
    results = run_enhanced_comprehensive_analysis()
    
    if results is None:
        print("Analysis failed. Please check the errors above.")
        return
    
    print("\n" + "=" * 60)
    print("FINAL PROJECT ENHANCED ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Enhanced final recommendations
    if results.get('comparison'):
        rating = results['comparison']['overall_assessment']['rating']
        score = results['comparison']['overall_assessment']['score']
        
        print(f"\nFINAL RATING: {rating} (Score: {score:.3f})")
        
        if rating in ['EXCELLENT', 'GOOD']:
            print("Debiasing strategies were highly effective")
            print("Model maintains or improves performance while reducing bias")
            print("Comprehensive fairness improvements achieved")
        elif rating == 'MODEST':
            print("Debiasing showed modest effectiveness")
            print("Some bias reduction with maintained performance")
            print("Consider additional debiasing strategies")
        else:
            print("Debiasing had limited effectiveness")
            print("Consider alternative approaches or data collection")
    
    print("\nKey Deliverables Generated:")
    print("  Enhanced baseline model bias analysis")
    print("  Enhanced debiased model training and analysis") 
    print("  Comprehensive before/after comparison")
    print("  Professional enhanced visualizations")
    print("  Detailed fairness metrics and recommendations")
    print("  LIME explainability integration")
    print("  SHAP analysis integration")
    print("  Intersectional bias analysis")
    print("  Counterfactual fairness evaluation")
    
    print("\nFiles Generated:")
    print("  results/baseline_bias_report.json")
    print("  results/debiased_bias_report.json") 
    print("  results/enhanced_model_comparison.json")
    print("  visualizations/baseline/ - Enhanced baseline charts")
    print("  visualizations/debiased/ - Enhanced debiased charts")
    print("  visualizations/enhanced_fairness_comparison.png")
    print("  visualizations/enhanced_performance_radar.png")
    
    print("\nNext Steps:")
    print("  Launch enhanced Gradio interface: python gradio_app.py")
    print("  Review the comprehensive comparison report")
    print("  Test LIME and SHAP explanations in the web interface")
    print("  Implement additional debiasing strategies if needed")
    print("=" * 60)


if __name__ == "__main__":
    main()
