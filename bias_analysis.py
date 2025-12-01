"""
Bias Analysis Script
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import pickle
import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bias_analyzer import EnhancedBiasAnalyzer, EnhancedDemographicInference
from sklearn.preprocessing import LabelEncoder


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types recursively"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
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


class AdvancedFairnessMetrics:
    """Advanced fairness metrics for multi-class classification"""
    
    @staticmethod
    def equalized_odds_difference(y_true, y_pred, protected_attr, num_classes):
        """
        Equalized Odds: Equal TPR and FPR across groups
        Returns maximum difference in TPR and FPR across groups
        """
        protected_attr = np.array(protected_attr)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        group_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                # Calculate TPR and FPR for each class
                tpr_list = []
                fpr_list = []
                
                for cls in range(num_classes):
                    # True positives for this class
                    tp = np.sum((y_true_group == cls) & (y_pred_group == cls))
                    fn = np.sum((y_true_group == cls) & (y_pred_group != cls))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tpr_list.append(tpr)
                    
                    # False positives for this class
                    fp = np.sum((y_true_group != cls) & (y_pred_group == cls))
                    tn = np.sum((y_true_group != cls) & (y_pred_group != cls))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fpr_list.append(fpr)
                
                group_metrics[group] = {
                    'tpr': np.mean(tpr_list),
                    'fpr': np.mean(fpr_list)
                }
        
        # Calculate differences
        tpr_values = [metrics['tpr'] for metrics in group_metrics.values()]
        fpr_values = [metrics['fpr'] for metrics in group_metrics.values()]
        
        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0
        
        return {
            'equalized_odds_difference': max(tpr_diff, fpr_diff),
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff
        }
    
    @staticmethod
    def treatment_equality_ratio(y_true, y_pred, protected_attr):
        """
        Treatment Equality: Ratio of false positive to false negative rates
        Should be equal across groups
        """
        protected_attr = np.array(protected_attr)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        group_ratios = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                # False positives and false negatives
                fp = np.sum((y_true_group != y_pred_group) & (y_pred_group != -1))
                fn = np.sum((y_true_group != y_pred_group) & (y_pred_group == -1))
                
                # Avoid division by zero
                ratio = fp / fn if fn > 0 else float('inf')
                group_ratios.append(ratio)
        
        # Normalize ratios to [0, 1] range
        valid_ratios = [r for r in group_ratios if r != float('inf')]
        if len(valid_ratios) >= 2:
            max_ratio = max(valid_ratios)
            min_ratio = min(valid_ratios)
            if max_ratio > 0:
                normalized_diff = (max_ratio - min_ratio) / max_ratio
            else:
                normalized_diff = 0
        else:
            normalized_diff = 0
        
        return normalized_diff
    
    @staticmethod
    def intersectional_fairness(y_true, y_pred, protected_attrs_dict):
        """
        Intersectional fairness across multiple protected attributes
        protected_attrs_dict: dictionary of protected attributes
        """
        protected_names = list(protected_attrs_dict.keys())
        protected_values = list(protected_attrs_dict.values())
        
        # Get all intersectional groups
        intersectional_groups = set(zip(*protected_values))
        
        group_accuracies = {}
        
        for combo in intersectional_groups:
            # Create mask for this intersectional group
            group_mask = np.ones(len(y_true), dtype=bool)
            for i, values in enumerate(protected_values):
                group_mask &= (np.array(values) == combo[i])
            
            if np.sum(group_mask) > 0:
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]
                
                accuracy = np.mean(y_true_group == y_pred_group)
                group_name = "_".join([f"{protected_names[i]}_{combo[i]}" 
                                      for i in range(len(combo))])
                group_accuracies[group_name] = {
                    'accuracy': accuracy,
                    'size': np.sum(group_mask)
                }
        
        # Calculate fairness score (1 - max accuracy gap)
        if group_accuracies:
            accuracies = [info['accuracy'] for info in group_accuracies.values()]
            fairness_score = 1 - (max(accuracies) - min(accuracies))
        else:
            fairness_score = 1
        
        return {
            'intersectional_fairness': fairness_score,
            'group_accuracies': group_accuracies,
            'min_accuracy': min(accuracies) if group_accuracies else 0,
            'max_accuracy': max(accuracies) if group_accuracies else 0,
            'accuracy_range': max(accuracies) - min(accuracies) if group_accuracies else 0
        }


def analyze_model(model_path, model_type, test_texts, test_labels, label_map):
    """Analyze bias for a specific model"""
    print(f"Analyzing {model_type} model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Get predictions
    predictions = []
    for text in test_texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1)
            predictions.append(pred.item())
    
    # Run bias analysis
    analyzer = EnhancedBiasAnalyzer(model, tokenizer, label_map, device)
    bias_report = analyzer.comprehensive_bias_analysis(test_texts, test_labels, predictions)
    
    # Add advanced fairness metrics
    print("Calculating advanced fairness metrics...")
    advanced_metrics = AdvancedFairnessMetrics()
    
    # Infer demographics for advanced metrics
    demo_inference = EnhancedDemographicInference()
    genders = [demo_inference.infer_gender(text) for text in test_texts]
    privileges = [demo_inference.infer_educational_privilege(text) for text in test_texts]
    
    # Encode categorical variables
    gender_encoder = LabelEncoder()
    privilege_encoder = LabelEncoder()
    gender_encoded = gender_encoder.fit_transform(genders)
    privilege_encoded = privilege_encoder.fit_transform(privileges)
    
    # Calculate advanced metrics
    num_classes = len(label_map)
    
    # Equalized Odds
    equalized_odds = advanced_metrics.equalized_odds_difference(
        np.array(test_labels),
        np.array(predictions),
        gender_encoded,
        num_classes
    )
    
    # Treatment Equality
    treatment_eq = advanced_metrics.treatment_equality_ratio(
        np.array(test_labels),
        np.array(predictions),
        gender_encoded
    )
    
    # Intersectional Fairness
    protected_dict = {
        'gender': genders,
        'privilege': privileges
    }
    intersectional = advanced_metrics.intersectional_fairness(
        np.array(test_labels),
        np.array(predictions),
        protected_dict
    )
    
    # Add to report
    bias_report['advanced_fairness_metrics'] = {
        'equalized_odds': equalized_odds,
        'treatment_equality': treatment_eq,
        'intersectional_fairness': intersectional
    }
    
    # Print advanced metrics summary
    print("\nADVANCED FAIRNESS METRICS:")
    print(f"Equalized Odds Difference: {equalized_odds.get('equalized_odds_difference', 0):.3f}")
    print(f"  TPR Difference: {equalized_odds.get('tpr_difference', 0):.3f}")
    print(f"  FPR Difference: {equalized_odds.get('fpr_difference', 0):.3f}")
    print(f"Treatment Equality Ratio: {treatment_eq:.3f}")
    print(f"Intersectional Fairness: {intersectional.get('intersectional_fairness', 0):.3f}")
    if 'min_accuracy' in intersectional:
        print(f"  Min Group Accuracy: {intersectional.get('min_accuracy', 0)*100:.1f}%")
        print(f"  Max Group Accuracy: {intersectional.get('max_accuracy', 0)*100:.1f}%")
        print(f"  Accuracy Range: {intersectional.get('accuracy_range', 0)*100:.1f}%")
    
    # Convert numpy types to Python native types
    bias_report_serializable = convert_to_serializable(bias_report)
    
    # Save report
    report_path = f'results/{model_type}_bias_report.json'
    with open(report_path, 'w') as f:
        json.dump(bias_report_serializable, f, indent=2, cls=NumpyEncoder)
    
    print(f"Bias report saved to {report_path}")
    return bias_report_serializable


def main():
    """Main bias analysis function"""
    print("Bias Analysis and Model Comparison")
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    
    # Load training data
    with open('data/processed/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    test_texts = data['X_test']
    test_labels = data['y_test']
    label_map = data['label_map']
    
    # Analyze baseline model
    baseline_report = analyze_model(
        'models/resume_classifier_baseline',
        'baseline',
        test_texts,
        test_labels,
        label_map
    )
    
    # Analyze debiased model
    debiased_report = analyze_model(
        'models/resume_classifier_debiased',
        'debiased',
        test_texts,
        test_labels,
        label_map
    )
    
    # Generate comparison
    comparison = {
        'performance': {
            'baseline_accuracy': float(baseline_report['overall_performance']['accuracy']),
            'debiased_accuracy': float(debiased_report['overall_performance']['accuracy']),
            'accuracy_change_percent': float((debiased_report['overall_performance']['accuracy'] - 
                                           baseline_report['overall_performance']['accuracy']) * 100)
        },
        'bias_reduction': {
            'gender_bias': {
                'baseline': float(baseline_report['name_substitution_bias']['gender_bias']['average_bias']),
                'debiased': float(debiased_report['name_substitution_bias']['gender_bias']['average_bias']),
                'reduction_percent': float((baseline_report['name_substitution_bias']['gender_bias']['average_bias'] - 
                                         debiased_report['name_substitution_bias']['gender_bias']['average_bias']) * 100)
            }
        },
        'fairness_improvement': {
            'gender': {
                'baseline_dp': float(baseline_report['fairness_metrics']['gender']['demographic_parity']),
                'debiased_dp': float(debiased_report['fairness_metrics']['gender']['demographic_parity']),
                'improvement': float(baseline_report['fairness_metrics']['gender']['demographic_parity'] - 
                                  debiased_report['fairness_metrics']['gender']['demographic_parity'])
            }
        }
    }
    
    with open('results/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, cls=NumpyEncoder)
    
    print("\nModel Comparison Results:")
    print(f"Baseline Accuracy: {comparison['performance']['baseline_accuracy']*100:.2f}%")
    print(f"Debiased Accuracy: {comparison['performance']['debiased_accuracy']*100:.2f}%")
    print(f"Accuracy Change: {comparison['performance']['accuracy_change_percent']:+.2f}%")
    print(f"Gender Bias Reduction: {comparison['bias_reduction']['gender_bias']['reduction_percent']:+.2f}%")
    
    # Save to a simpler file for Gradio
    with open('results/enhanced_model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
