"""
Bias Analysis Script - Run after training completes
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_trainer import ResumeDataset
from bias_analyzer import BiasAnalyzer, BiasVisualization


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


def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        model_path = 'models/resume_classifier'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print("Model and tokenizer loaded successfully")
        return model, tokenizer, device
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please run 'python train.py' first to train the model")
        return None, None, None


def run_comprehensive_bias_analysis():
    """Run comprehensive bias analysis on trained model"""
    print("=" * 60)
    print("COMPREHENSIVE BIAS ANALYSIS")
    print("=" * 60)
    print("CAI 6605: Trustworthy AI Systems - Final Project")
    print("Group 15: Nithin Palyam, Lorenzo LaPlace")
    print("=" * 60)
    
    # Load training data
    training_data = load_training_data()
    if training_data is None:
        return None
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    if model is None:
        return None
    
    # Extract data
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    label_map = training_data['label_map']
    
    # Create test dataset
    test_dataset = ResumeDataset(X_test, y_test, tokenizer, 512)
    
    # Get predictions for bias analysis
    class SimpleTrainer:
        def __init__(self, model):
            self.model = model
        
        def predict(self, dataset):
            self.model.eval()
            all_predictions = []
            
            with torch.no_grad():
                for i in range(len(dataset)):
                    inputs = dataset[i]
                    input_ids = inputs['input_ids'].unsqueeze(0).to(device)
                    attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    all_predictions.append(outputs.logits.cpu().numpy())
            
            return type('obj', (object,), {
                'predictions': np.vstack(all_predictions)
            })
    
    print("Getting model predictions...")
    trainer = SimpleTrainer(model)
    test_predictions = trainer.predict(test_dataset)
    test_pred_labels = np.argmax(test_predictions.predictions, axis=1)
    
    # Initialize bias analyzer
    bias_analyzer = BiasAnalyzer(model, tokenizer, label_map, device)
    
    # Run comprehensive bias analysis
    bias_report = bias_analyzer.comprehensive_bias_analysis(
        X_test, y_test, test_pred_labels
    )
    
    # Generate visualizations
    print("\nGenerating bias visualizations...")
    try:
        BiasVisualization.plot_fairness_metrics(
            bias_report['fairness_metrics'],
            save_path='visualizations/fairness_metrics.png'
        )
        
        BiasVisualization.plot_category_bias(
            bias_report['category_bias_analysis'],
            save_path='visualizations/category_bias.png'
        )
        print("Visualizations saved to 'visualizations/' directory")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
    
    # Save comprehensive bias report
    with open('results/comprehensive_bias_report.json', 'w') as f:
        json.dump(bias_report, f, indent=2)
    
    print("Bias analysis complete! Report saved to 'results/comprehensive_bias_report.json'")
    
    return bias_report


def main():
    """Main bias analysis pipeline"""
    # Run comprehensive bias analysis
    bias_report = run_comprehensive_bias_analysis()
    
    if bias_report is None:
        return
    
    # Final summary
    print("\n" + "=" * 60)
    print("BIAS ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Bias analysis summary
    avg_fairness_metrics = {}
    for demo_type, metrics in bias_report['fairness_metrics'].items():
        avg_fairness = np.mean([metrics['demographic_parity'], metrics['equal_opportunity'], metrics['accuracy_equality']])
        avg_fairness_metrics[demo_type] = avg_fairness
    
    print(f"BIAS ANALYSIS SUMMARY:")
    for demo_type, fairness_score in avg_fairness_metrics.items():
        status = "GOOD" if fairness_score < 0.1 else "NEEDS ATTENTION"
        print(f"  {demo_type.upper():20s} Fairness Score: {fairness_score:.3f} ({status})")
    
    name_bias_data = bias_report['name_substitution_bias']
    print(f"NAME-BASED BIAS: {name_bias_data['average_gender_bias']:.3f}")
    
    print("\nKey Achievements:")
    print("  Comprehensive bias detection framework")
    print("  Demographic parity and equal opportunity metrics")
    print("  Name substitution experiments for bias measurement")
    print("  Category-level bias analysis across job types")
    print("  Professional visualizations and reporting")
    
    print("\nNext steps:")
    print("Launch web interface: python gradio_app.py")
    print("View visualizations in 'visualizations/' directory")
    print("Check detailed report: 'results/comprehensive_bias_report.json'")
    print("=" * 60)
    
    return bias_report


if __name__ == "__main__":
    main()
