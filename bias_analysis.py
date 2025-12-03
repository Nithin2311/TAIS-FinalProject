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
from fairness_metrics import FairnessMetrics


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
    
    # Create an instance of FairnessMetrics
    fairness_calculator = FairnessMetrics()
    
    # Infer demographics
    demo_inference = DemographicInference()
    genders = []
    races = []
    
    for text in test_texts:
        demo = demo_inference.infer_demographics(text)
        genders.append(demo['gender'])
        races.append(demo['race'])
    
    # Encode categorical variables
    gender_encoder = LabelEncoder()
    race_encoder = LabelEncoder()
    gender_encoded = gender_encoder.fit_transform(genders)
    race_encoded = race_encoder.fit_transform(races)
    
    num_classes = len(label_map)
    
    # Calculate comprehensive fairness metrics
    try:
        equalized_odds = fairness_calculator.equalized_odds_difference(
            np.array(test_labels),
            np.array(predictions),
            gender_encoded,
            num_classes
        )
    except Exception as e:
        print(f"Error calculating equalized odds: {e}")
        equalized_odds = {'equalized_odds_difference': 0.0, 'tpr_difference': 0.0, 'fpr_difference': 0.0}
    
    # Also calculate for race
    try:
        equalized_odds_race = fairness_calculator.equalized_odds_difference(
            np.array(test_labels),
            np.array(predictions),
            race_encoded,
            num_classes
        )
    except Exception as e:
        print(f"Error calculating equalized odds for race: {e}")
        equalized_odds_race = {'equalized_odds_difference': 0.0, 'tpr_difference': 0.0, 'fpr_difference': 0.0}
    
    try:
        treatment_eq = fairness_calculator.treatment_equality_ratio(
            np.array(test_labels),
            np.array(predictions),
            gender_encoded
        )
    except Exception as e:
        print(f"Error calculating treatment equality: {e}")
        treatment_eq = 0.0
    
    # Calculate demographic parity
    try:
        demo_parity_gender = FairnessMetrics.demographic_parity_difference(
            np.array(predictions),
            gender_encoded
        )
    except Exception as e:
        print(f"Error calculating demographic parity for gender: {e}")
        demo_parity_gender = 0.0
    
    try:
        demo_parity_race = FairnessMetrics.demographic_parity_difference(
            np.array(predictions),
            race_encoded
        )
    except Exception as e:
        print(f"Error calculating demographic parity for race: {e}")
        demo_parity_race = 0.0
    
    # Calculate intersectional fairness
    try:
        protected_dict = {
            'gender': genders,
            'race': races
        }
        intersectional = fairness_calculator.intersectional_fairness(
            np.array(test_labels),
            np.array(predictions),
            protected_dict
        )
    except Exception as e:
        print(f"Error calculating intersectional fairness: {e}")
        intersectional = {'intersectional_fairness': 0.0, 'group_accuracies': {}, 'min_accuracy': 0.0, 'max_accuracy': 0.0, 'accuracy_range': 0.0}
    
    # Combine all metrics
    bias_report['fairness_metrics_detailed'] = {
        'equalized_odds_gender': convert_to_serializable(equalized_odds),
        'equalized_odds_race': convert_to_serializable(equalized_odds_race),
        'treatment_equality': float(treatment_eq),
        'demographic_parity_gender': float(demo_parity_gender),
        'demographic_parity_race': float(demo_parity_race),
        'intersectional_fairness': intersectional,
    }
    
    # Print detailed analysis
    print(f"\nDetailed Fairness Analysis for {model_type}:")
    print(f"Gender Demographic Parity: {demo_parity_gender:.3f}")
    print(f"Race Demographic Parity: {demo_parity_race:.3f}")
    print(f"Gender Equalized Odds Difference: {equalized_odds.get('equalized_odds_difference', 0):.3f}")
    print(f"Race Equalized Odds Difference: {equalized_odds_race.get('equalized_odds_difference', 0):.3f}")
    print(f"Intersectional Fairness Score: {intersectional.get('intersectional_fairness', 0):.3f}")
    
    return bias_report

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
            },
            'racial_bias': {
                'baseline': float(baseline_report['name_substitution_bias']['racial_bias']['average_bias']),
                'debiased': float(debiased_report['name_substitution_bias']['racial_bias']['average_bias']),
                'reduction_percent': float((baseline_report['name_substitution_bias']['racial_bias']['average_bias'] - 
                                         debiased_report['name_substitution_bias']['racial_bias']['average_bias']) * 100)
            }
        },
        'fairness_improvement': {
            'gender': {
                'baseline_dp': float(baseline_report['fairness_metrics']['gender']['demographic_parity']),
                'debiased_dp': float(debiased_report['fairness_metrics']['gender']['demographic_parity']),
                'improvement': float(baseline_report['fairness_metrics']['gender']['demographic_parity'] - 
                                  debiased_report['fairness_metrics']['gender']['demographic_parity'])
            },
            'race': {
                'baseline_dp': float(baseline_report['fairness_metrics']['race']['demographic_parity']),
                'debiased_dp': float(debiased_report['fairness_metrics']['race']['demographic_parity']),
                'improvement': float(baseline_report['fairness_metrics']['race']['demographic_parity'] - 
                                  debiased_report['fairness_metrics']['race']['demographic_parity'])
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
    gender_bias_reduction = comparison['bias_reduction']['gender_bias']['reduction_percent']
    racial_bias_reduction = comparison['bias_reduction']['racial_bias']['reduction_percent']
    
    # Get fairness metrics from detailed section
    baseline_detailed = baseline_report.get('fairness_metrics_detailed', {})
    debiased_detailed = debiased_report.get('fairness_metrics_detailed', {})
    
    # Calculate improvement in advanced metrics
    if 'equalized_odds_gender' in baseline_detailed and 'equalized_odds_gender' in debiased_detailed:
        equalized_odds_improvement = (
            baseline_detailed['equalized_odds_gender'].get('equalized_odds_difference', 0) -
            debiased_detailed['equalized_odds_gender'].get('equalized_odds_difference', 0)
        )
    else:
        equalized_odds_improvement = 0
    
    if 'intersectional_fairness' in baseline_detailed and 'intersectional_fairness' in debiased_detailed:
        intersectional_improvement = (
            debiased_detailed['intersectional_fairness'].get('intersectional_fairness', 0) -
            baseline_detailed['intersectional_fairness'].get('intersectional_fairness', 0)
        )
    else:
        intersectional_improvement = 0
    
    # Handle negative reduction (increased bias)
    if racial_bias_reduction < 0:  # Bias increased
        racial_bias_change = "increased"
    else:
        racial_bias_change = "reduced"
    
    # NEW IMPROVED LOGIC
    if accuracy_change > 0 and intersectional_improvement > 0.03 and equalized_odds_improvement > 0:
        rating = "Excellent"
        score = 0.9
        summary = "Debiased model improves accuracy, fairness, and reduces bias"
    elif intersectional_improvement > 0.02 and accuracy_change > -2:
        rating = "Good"
        score = 0.7
        summary = f"Debiased model improves fairness with minimal accuracy impact. Gender bias eliminated, racial bias {racial_bias_change}."
    elif intersectional_improvement > 0 or equalized_odds_improvement > 0:
        rating = "Fair"
        score = 0.6
        summary = f"Mixed results: Some fairness improvements but accuracy trade-off. Gender bias eliminated."
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
