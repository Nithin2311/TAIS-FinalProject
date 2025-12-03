"""
Bias analysis and model comparison script.
"""

import pickle
import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bias_analyzer import BiasAnalyzer, DemographicInference
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
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
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


class FairnessMetrics:
    """Fairness metrics for multi-class classification"""
    
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
                
                tpr_list = []
                fpr_list = []
                
                for cls in range(num_classes):
                    tp = np.sum((y_true_group == cls) & (y_pred_group == cls))
                    fn = np.sum((y_true_group == cls) & (y_pred_group != cls))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tpr_list.append(tpr)
                    
                    fp = np.sum((y_true_group != cls) & (y_pred_group == cls))
                    tn = np.sum((y_true_group != cls) & (y_pred_group != cls))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fpr_list.append(fpr)
                
                group_metrics[group] = {
                    'tpr': np.mean(tpr_list),
                    'fpr': np.mean(fpr_list)
                }
        
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
                
                fp = np.sum((y_true_group != y_pred_group) & (y_pred_group != -1))
                fn = np.sum((y_true_group != y_pred_group) & (y_pred_group == -1))
                
                ratio = fp / fn if fn > 0 else float('inf')
                group_ratios.append(ratio)
        
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
        
        intersectional_groups = set(zip(*protected_values))
        
        group_accuracies = {}
        
        for combo in intersectional_groups:
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
                    'accuracy': float(accuracy),
                    'size': int(np.sum(group_mask))
                }
        
        if group_accuracies:
            accuracies = [info['accuracy'] for info in group_accuracies.values()]
            fairness_score = 1 - (max(accuracies) - min(accuracies))
        else:
            fairness_score = 1
        
        return {
            'intersectional_fairness': float(fairness_score),
            'group_accuracies': group_accuracies,
            'min_accuracy': float(min(accuracies)) if group_accuracies else 0,
            'max_accuracy': float(max(accuracies)) if group_accuracies else 0,
            'accuracy_range': float(max(accuracies) - min(accuracies)) if group_accuracies else 0
        }


def analyze_model(model_path, model_type, test_texts, test_labels, label_map):
    """Analyze bias for a specific model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        else:
            alt_paths = [
                model_path,
                f"./{model_path}",
                f"models/{model_type}_model",
                f"models/resume_classifier_{model_type}"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    tokenizer = AutoTokenizer.from_pretrained(alt_path, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(alt_path, local_files_only=True)
                    break
            else:
                print(f"Could not find {model_type} model.")
                return None
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None
    
    model.to(device)
    model.eval()
    
    predictions = []
    batch_size = 16
    
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        try:
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(batch_preds.cpu().numpy().tolist())
                
        except Exception as e:
            print(f"Error predicting batch {i//batch_size}: {e}")
            predictions.extend([0] * len(batch_texts))
    
    analyzer = BiasAnalyzer(model, tokenizer, label_map, device)
    bias_report = analyzer.bias_analysis(test_texts, test_labels, predictions)
    
    fairness_metrics = FairnessMetrics()
    
    demo_inference = DemographicInference()
    genders = [demo_inference.infer_gender(text) for text in test_texts]
    races = [demo_inference.infer_race_from_names(text) for text in test_texts]
    
    gender_encoder = LabelEncoder()
    race_encoder = LabelEncoder()
    gender_encoded = gender_encoder.fit_transform(genders)
    race_encoded = race_encoder.fit_transform(races)
    
    num_classes = len(label_map)
    
    equalized_odds = fairness_metrics.equalized_odds_difference(
        np.array(test_labels),
        np.array(predictions),
        gender_encoded,
        num_classes
    )
    
    treatment_eq = fairness_metrics.treatment_equality_ratio(
        np.array(test_labels),
        np.array(predictions),
        gender_encoded
    )
    
    protected_dict = {
        'gender': genders,
        'race': races
    }
    intersectional = fairness_metrics.intersectional_fairness(
        np.array(test_labels),
        np.array(predictions),
        protected_dict
    )
    
    bias_report['advanced_fairness_metrics'] = {
        'equalized_odds': convert_to_serializable(equalized_odds),
        'treatment_equality': float(treatment_eq),
        'intersectional_fairness': intersectional
    }
    
    print(f"\nAdvanced Fairness Metrics for {model_type}:")
    print(f"Equalized Odds Difference: {equalized_odds.get('equalized_odds_difference', 0):.3f}")
    print(f"  TPR Difference: {equalized_odds.get('tpr_difference', 0):.3f}")
    print(f"  FPR Difference: {equalized_odds.get('fpr_difference', 0):.3f}")
    print(f"Treatment Equality Ratio: {treatment_eq:.3f}")
    print(f"Intersectional Fairness: {intersectional.get('intersectional_fairness', 0):.3f}")
    if 'min_accuracy' in intersectional:
        print(f"  Min Group Accuracy: {intersectional.get('min_accuracy', 0)*100:.1f}%")
        print(f"  Max Group Accuracy: {intersectional.get('max_accuracy', 0)*100:.1f}%")
        print(f"  Accuracy Range: {intersectional.get('accuracy_range', 0)*100:.1f}%")
    
    bias_report_serializable = convert_to_serializable(bias_report)
    
    report_path = f'results/{model_type}_bias_report.json'
    with open(report_path, 'w') as f:
        json.dump(bias_report_serializable, f, indent=2, cls=NumpyEncoder)
    
    print(f"Bias report saved to {report_path}")
    return bias_report_serializable


def compare_models(baseline_report, debiased_report):
    """Compare baseline and debiased models"""
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
    
    baseline_accuracies = baseline_report.get('category_bias_analysis', {})
    debiased_accuracies = debiased_report.get('category_bias_analysis', {})
    
    if baseline_accuracies and debiased_accuracies:
        category_improvements = []
        for category in baseline_accuracies:
            if category in debiased_accuracies:
                baseline_acc = baseline_accuracies[category].get('overall_accuracy', 0)
                debiased_acc = debiased_accuracies[category].get('overall_accuracy', 0)
                if baseline_acc > 0:
                    improvement = (debiased_acc - baseline_acc) / baseline_acc * 100
                    category_improvements.append((category, improvement))
        
        category_improvements.sort(key=lambda x: x[1], reverse=True)
        most_improved = category_improvements[:5]
        
        comparison['category_improvements'] = {
            'most_improved': [(cat, float(imp)) for cat, imp in most_improved],
            'total_improved': len([imp for _, imp in category_improvements if imp > 0]),
            'avg_improvement': float(np.mean([imp for _, imp in category_improvements])) if category_improvements else 0
        }
    
    accuracy_change = comparison['performance']['accuracy_change_percent']
    bias_reduction = comparison['bias_reduction']['gender_bias']['reduction_percent']
    
    if accuracy_change > 0 and bias_reduction > 0:
        rating = "Excellent"
        score = 0.9
        summary = "Debiased model improves both accuracy and fairness"
    elif bias_reduction > 0 and accuracy_change > -5:
        rating = "Good"
        score = 0.7
        summary = "Debiased model reduces bias with minimal accuracy trade-off"
    elif bias_reduction > 0:
        rating = "Fair"
        score = 0.6
        summary = "Debiased model reduces bias but with some accuracy loss"
    else:
        rating = "Needs Improvement"
        score = 0.4
        summary = "Further bias mitigation needed"
    
    comparison['overall_assessment'] = {
        'rating': rating,
        'score': score,
        'summary': summary
    }
    
    return comparison


def main():
    """Main bias analysis function"""
    print("Bias Analysis and Model Comparison")
    
    if not os.path.exists('data/processed/training_data.pkl'):
        print("Error: Training data not found. Please run train_baseline.py first.")
        return
    
    with open('data/processed/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    test_texts = data['X_test']
    test_labels = data['y_test']
    label_map = data['label_map']
    
    label_map_str = {str(k): v for k, v in label_map.items()}
    
    model_paths = {
        'baseline': 'models/resume_classifier_baseline',
        'debiased': 'models/resume_classifier_debiased'
    }
    
    available_models = {}
    for model_type, model_path in model_paths.items():
        if os.path.exists(model_path):
            available_models[model_type] = model_path
            print(f"Found {model_type} model at {model_path}")
        else:
            print(f"Warning: {model_type} model not found at {model_path}")
    
    if not available_models:
        print("No models found. Please train at least one model first.")
        return
    
    reports = {}
    for model_type, model_path in available_models.items():
        report = analyze_model(model_path, model_type, test_texts, test_labels, label_map_str)
        if report is not None:
            reports[model_type] = report
    
    if len(reports) < 2:
        print(f"\nOnly {len(reports)} model(s) analyzed. Skipping comparison.")
        return
    
    if 'baseline' in reports and 'debiased' in reports:
        baseline_report = reports['baseline']
        debiased_report = reports['debiased']
        
        comparison = compare_models(baseline_report, debiased_report)
        
        with open('results/model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, cls=NumpyEncoder)
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(f"Baseline Accuracy: {comparison['performance']['baseline_accuracy']*100:.2f}%")
        print(f"Debiased Accuracy: {comparison['performance']['debiased_accuracy']*100:.2f}%")
        print(f"Accuracy Change: {comparison['performance']['accuracy_change_percent']:+.2f}%")
        print(f"Gender Bias Reduction: {comparison['bias_reduction']['gender_bias']['reduction_percent']:+.2f}%")
        
        assessment = comparison['overall_assessment']
        print(f"\nOverall Assessment: {assessment['rating']}")
        print(f"Score: {assessment['score']:.1f}/1.0")
        print(f"Summary: {assessment['summary']}")
        
        with open('results/simplified_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, cls=NumpyEncoder)
    
    print("\nBias analysis completed successfully.")


if __name__ == "__main__":
    main()
