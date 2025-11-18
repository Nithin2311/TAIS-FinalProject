"""
Advanced Bias Detection and Fairness Analysis Module
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DemographicInference:
    """Infer demographic attributes from resume text"""
    
    def __init__(self):
        # Gender inference patterns
        self.gender_keywords = {
            'male': ['he', 'him', 'his', 'male', 'man', 'boy', 'mr', 'mister'],
            'female': ['she', 'her', 'hers', 'female', 'woman', 'girl', 'ms', 'miss', 'mrs']
        }
        
        # Diversity background inference
        self.diversity_keywords = {
            'underrepresented': [
                'first generation', 'low-income', 'underserved', 'minority',
                'diversity scholarship', 'affirmative action', 'equity program'
            ],
            'privileged': [
                'legacy', 'ivy league', 'prep school', 'private school',
                'summer abroad', 'study abroad', 'exchange program'
            ]
        }
        
        # Name-based inference (simplified)
        self.common_names = {
            'male': ['james', 'john', 'robert', 'michael', 'william', 'david', 'richard'],
            'female': ['mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'barbara'],
            'diverse': ['lakisha', 'jamal', 'tyrone', 'latoya', 'juan', 'maria', 'wei', 'li']
        }
    
    def infer_gender(self, text):
        """Infer gender from text using keyword analysis"""
        text_lower = text.lower()
        
        male_count = sum(text_lower.count(word) for word in self.gender_keywords['male'])
        female_count = sum(text_lower.count(word) for word in self.gender_keywords['female'])
        
        if male_count > female_count:
            return 'male'
        elif female_count > male_count:
            return 'female'
        else:
            return 'unknown'
    
    def infer_diversity_background(self, text):
        """Infer diversity background from text"""
        text_lower = text.lower()
        
        underrepresented_count = sum(text_lower.count(phrase) for phrase in self.diversity_keywords['underrepresented'])
        privileged_count = sum(text_lower.count(phrase) for phrase in self.diversity_keywords['privileged'])
        
        if underrepresented_count > privileged_count:
            return 'underrepresented'
        elif privileged_count > underrepresented_count:
            return 'privileged'
        else:
            return 'neutral'
    
    def infer_privilege_level(self, text):
        """Infer privilege level based on education and experience indicators"""
        text_lower = text.lower()
        
        elite_indicators = [
            'harvard', 'stanford', 'mit', 'princeton', 'yale', 'columbia',
            'mckinsey', 'goldman sachs', 'google', 'microsoft', 'apple'
        ]
        
        elite_count = sum(text_lower.count(indicator) for indicator in elite_indicators)
        
        if elite_count >= 3:
            return 'high'
        elif elite_count >= 1:
            return 'medium'
        else:
            return 'low'


class FairnessMetrics:
    """Compute comprehensive fairness metrics"""
    
    @staticmethod
    def demographic_parity_difference(y_true, y_pred, protected_attr):
        """Calculate demographic parity difference"""
        groups = np.unique(protected_attr)
        positive_rates = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred[group_mask])
                positive_rates.append(positive_rate)
        
        return max(positive_rates) - min(positive_rates) if positive_rates else 0
    
    @staticmethod
    def equal_opportunity_difference(y_true, y_pred, protected_attr):
        """Calculate equal opportunity difference (recall disparity)"""
        groups = np.unique(protected_attr)
        recall_rates = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask & (y_true == 1)) > 0:
                recall = np.sum(y_pred[group_mask & (y_true == 1)]) / np.sum(group_mask & (y_true == 1))
                recall_rates.append(recall)
        
        return max(recall_rates) - min(recall_rates) if recall_rates else 0
    
    @staticmethod
    def disparate_impact_ratio(y_true, y_pred, protected_attr, privileged_group):
        """Calculate disparate impact ratio"""
        groups = np.unique(protected_attr)
        positive_rates = {}
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                positive_rates[group] = np.mean(y_pred[group_mask])
        
        if privileged_group in positive_rates and len(positive_rates) > 1:
            privileged_rate = positive_rates[privileged_group]
            unprivileged_rates = [rate for group, rate in positive_rates.items() if group != privileged_group]
            return min(unprivileged_rates) / privileged_rate if privileged_rate > 0 else 1
        return 1
    
    @staticmethod
    def accuracy_equality_difference(y_true, y_pred, protected_attr):
        """Calculate accuracy equality difference"""
        groups = np.unique(protected_attr)
        accuracies = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
                accuracies.append(accuracy)
        
        return max(accuracies) - min(accuracies) if accuracies else 0


class NameSubstitutionExperiment:
    """Conduct name substitution experiments to measure name-based bias"""
    
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
    def substitute_names(self, text, original_name, new_name):
        """Replace names in text for bias testing"""
        # Simple substitution - in practice, use NER for better accuracy
        return text.replace(original_name, new_name)
    
    def run_gender_bias_experiment(self, test_texts, test_labels):
        """Measure gender bias through name substitution"""
        print("\n🧪 Running Gender Bias Experiment...")
        
        male_names = ['James', 'Robert', 'John', 'Michael']
        female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda']
        
        results = {
            'male_predictions': [],
            'female_predictions': [],
            'bias_scores': []
        }
        
        for i, (text, label) in enumerate(zip(test_texts[:100], test_labels[:100])):  # Sample for efficiency
            # Add placeholder name
            base_text = f"Candidate: [NAME] {text}"
            
            # Test with male name
            male_text = base_text.replace("[NAME]", np.random.choice(male_names))
            male_pred = self._predict_single(male_text)
            
            # Test with female name
            female_text = base_text.replace("[NAME]", np.random.choice(female_names))
            female_pred = self._predict_single(female_text)
            
            results['male_predictions'].append(male_pred)
            results['female_predictions'].append(female_pred)
            
            # Calculate bias score (difference in predictions)
            bias_score = abs(male_pred - female_pred)
            results['bias_scores'].append(bias_score)
        
        return results
    
    def _predict_single(self, text):
        """Make prediction for single text"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs[0].max().item()


class BiasAnalyzer:
    """Comprehensive bias analysis framework"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.demographic_inference = DemographicInference()
        self.fairness_metrics = FairnessMetrics()
        self.name_experiment = NameSubstitutionExperiment(tokenizer, model, device)
        
    def comprehensive_bias_analysis(self, test_texts, test_labels, test_predictions):
        """Run comprehensive bias analysis"""
        print("\n" + "=" * 70)
        print("🔍 COMPREHENSIVE BIAS ANALYSIS")
        print("=" * 70)
        
        # Infer demographics
        print("📊 Inferring demographic attributes...")
        demographics = self._infer_demographics(test_texts)
        
        # Calculate fairness metrics
        print("📈 Calculating fairness metrics...")
        fairness_results = self._calculate_fairness_metrics(test_labels, test_predictions, demographics)
        
        # Category-level bias analysis
        print("🎯 Analyzing category-level bias...")
        category_bias = self._analyze_category_bias(test_labels, test_predictions, demographics)
        
        # Name substitution experiments
        print("👤 Running name substitution experiments...")
        name_bias = self.name_experiment.run_gender_bias_experiment(test_texts, test_labels)
        
        # Generate comprehensive report
        report = self._generate_bias_report(fairness_results, category_bias, name_bias)
        
        return report
    
    def _infer_demographics(self, texts):
        """Infer demographic attributes for all texts"""
        demographics = {
            'gender': [],
            'diversity_background': [],
            'privilege_level': []
        }
        
        for text in texts:
            demographics['gender'].append(self.demographic_inference.infer_gender(text))
            demographics['diversity_background'].append(
                self.demographic_inference.infer_diversity_background(text)
            )
            demographics['privilege_level'].append(
                self.demographic_inference.infer_privilege_level(text)
            )
        
        return demographics
    
    def _calculate_fairness_metrics(self, y_true, y_pred, demographics):
        """Calculate comprehensive fairness metrics"""
        metrics = {}
        
        for demo_type, demo_values in demographics.items():
            # Convert to numerical for metric calculation
            demo_encoder = LabelEncoder()
            demo_encoded = demo_encoder.fit_transform(demo_values)
            
            # Binary classification metrics (using top class as positive)
            y_true_binary = (y_true == np.argmax(np.bincount(y_true))).astype(int)
            y_pred_binary = (y_pred == np.argmax(np.bincount(y_pred))).astype(int)
            
            metrics[demo_type] = {
                'demographic_parity': self.fairness_metrics.demographic_parity_difference(
                    y_true_binary, y_pred_binary, demo_encoded
                ),
                'equal_opportunity': self.fairness_metrics.equal_opportunity_difference(
                    y_true_binary, y_pred_binary, demo_encoded
                ),
                'accuracy_equality': self.fairness_metrics.accuracy_equality_difference(
                    y_true, y_pred, demo_encoded
                )
            }
        
        return metrics
    
    def _analyze_category_bias(self, y_true, y_pred, demographics):
        """Analyze bias across different job categories"""
        category_bias = {}
        
        for category in range(len(self.label_map)):
            category_mask = y_true == category
            if np.sum(category_mask) > 10:  # Only analyze categories with sufficient samples
                category_accuracy = accuracy_score(y_true[category_mask], y_pred[category_mask])
                
                # Analyze demographic distribution in predictions
                demo_analysis = {}
                for demo_type, demo_values in demographics.items():
                    demo_encoder = LabelEncoder()
                    demo_encoded = demo_encoder.fit_transform(demo_values)
                    groups = demo_encoder.classes_
                    
                    group_accuracies = {}
                    for group in groups:
                        group_mask = np.array(demo_values) == group
                        combined_mask = category_mask & group_mask
                        if np.sum(combined_mask) > 5:
                            group_accuracy = accuracy_score(
                                y_true[combined_mask], 
                                y_pred[combined_mask]
                            )
                            group_accuracies[str(group)] = group_accuracy
                    
                    demo_analysis[demo_type] = group_accuracies
                
                category_bias[self.label_map[str(category)]] = {
                    'overall_accuracy': category_accuracy,
                    'demographic_analysis': demo_analysis
                }
        
        return category_bias
    
    def _generate_bias_report(self, fairness_results, category_bias, name_bias):
        """Generate comprehensive bias analysis report"""
        report = {
            'fairness_metrics': fairness_results,
            'category_bias_analysis': category_bias,
            'name_substitution_bias': {
                'average_gender_bias': np.mean(name_bias['bias_scores']),
                'male_female_disparity': np.mean(name_bias['male_predictions']) - np.mean(name_bias['female_predictions'])
            },
            'recommendations': self._generate_recommendations(fairness_results, category_bias, name_bias)
        }
        
        # Print summary
        self._print_bias_summary(report)
        
        return report
    
    def _generate_recommendations(self, fairness_results, category_bias, name_bias):
        """Generate actionable recommendations for bias mitigation"""
        recommendations = []
        
        # Analyze fairness metrics
        for demo_type, metrics in fairness_results.items():
            if metrics['demographic_parity'] > 0.1:
                recommendations.append(
                    f"High demographic parity disparity ({metrics['demographic_parity']:.3f}) for {demo_type}. "
                    f"Consider preprocessing techniques to balance representation."
                )
            
            if metrics['equal_opportunity'] > 0.1:
                recommendations.append(
                    f"Equal opportunity concerns ({metrics['equal_opportunity']:.3f}) for {demo_type}. "
                    f"Implement in-processing adversarial debiasing."
                )
        
        # Name bias recommendations
        if name_bias['average_gender_bias'] > 0.05:
            recommendations.append(
                f"Significant name-based gender bias detected ({name_bias['average_gender_bias']:.3f}). "
                f"Remove names during preprocessing and use post-processing calibration."
            )
        
        return recommendations
    
    def _print_bias_summary(self, report):
        """Print comprehensive bias analysis summary"""
        print("\n📋 BIAS ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Fairness metrics
        print("\n🎯 FAIRNESS METRICS:")
        for demo_type, metrics in report['fairness_metrics'].items():
            print(f"  {demo_type.upper():20s} | "
                  f"DemPar: {metrics['demographic_parity']:.3f} | "
                  f"EqOpp: {metrics['equal_opportunity']:.3f} | "
                  f"AccEq: {metrics['accuracy_equality']:.3f}")
        
        # Name bias
        name_bias = report['name_substitution_bias']
        print(f"\n👤 NAME-BASED BIAS:")
        print(f"  Average Gender Bias: {name_bias['average_gender_bias']:.3f}")
        print(f"  Male-Female Disparity: {name_bias['male_female_disparity']:.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        print("=" * 50)


class BiasVisualization:
    """Generate visualizations for bias analysis"""
    
    @staticmethod
    def plot_fairness_metrics(fairness_results, save_path=None):
        """Plot fairness metrics across demographic groups"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_to_plot = ['demographic_parity', 'equal_opportunity', 'accuracy_equality']
        titles = ['Demographic Parity Difference', 'Equal Opportunity Difference', 'Accuracy Equality Difference']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            demo_types = list(fairness_results.keys())
            values = [fairness_results[demo][metric] for demo in demo_types]
            
            axes[idx].bar(demo_types, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            axes[idx].set_title(title)
            axes[idx].set_ylabel('Disparity Score')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add threshold line
            axes[idx].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Fairness Threshold')
            axes[idx].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_category_bias(category_bias, save_path=None):
        """Plot bias across job categories"""
        categories = list(category_bias.keys())
        accuracies = [cat_data['overall_accuracy'] for cat_data in category_bias.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(categories, accuracies, color='skyblue', alpha=0.7)
        
        # Color bars based on performance
        for i, accuracy in enumerate(accuracies):
            if accuracy < 0.7:
                bars[i].set_color('red')
            elif accuracy < 0.8:
                bars[i].set_color('orange')
        
        plt.title('Classification Accuracy Across Job Categories')
        plt.xlabel('Job Categories')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
