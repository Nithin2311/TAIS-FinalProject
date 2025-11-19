"""
Enhanced Bias Detection and Fairness Analysis Module
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
from collections import defaultdict, Counter
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EnhancedDemographicInference:
    """Enhanced demographic inference from resume text"""
    
    def __init__(self):
        # Enhanced gender inference with more patterns
        self.gender_patterns = {
            'male': [
                r'\bhe\b', r'\bhim\b', r'\bhis\b', r'\bmale\b', r'\bman\b', r'\bmen\b',
                r'\bboy\b', r'\bmr\.', r'\bmr\b', r'\bmister\b'
            ],
            'female': [
                r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bfemale\b', r'\bwoman\b', r'\bwomen\b',
                r'\bgirl\b', r'\bms\.', r'\bms\b', r'\bmiss\b', r'\bmrs\.', r'\bmrs\b'
            ]
        }
        
        # Educational privilege indicators
        self.privilege_indicators = {
            'elite_universities': [
                'harvard', 'stanford', 'mit', 'princeton', 'yale', 'columbia',
                'cambridge', 'oxford', 'caltech', 'cornell'
            ],
            'prestigious_companies': [
                'google', 'microsoft', 'apple', 'amazon', 'meta', 'goldman sachs',
                'mckinsey', 'bain', 'boston consulting', 'jpmorgan'
            ]
        }
    
    def infer_gender(self, text):
        """Enhanced gender inference using multiple patterns"""
        text_lower = text.lower()
        
        male_score = 0
        female_score = 0
        
        for pattern in self.gender_patterns['male']:
            male_score += len(re.findall(pattern, text_lower))
        
        for pattern in self.gender_patterns['female']:
            female_score += len(re.findall(pattern, text_lower))
        
        if male_score > female_score:
            return 'male'
        elif female_score > male_score:
            return 'female'
        else:
            return 'unknown'
    
    def infer_educational_privilege(self, text):
        """Infer educational privilege based on institution mentions"""
        text_lower = text.lower()
        
        privilege_score = 0
        for university in self.privilege_indicators['elite_universities']:
            if university in text_lower:
                privilege_score += 2
        
        for company in self.privilege_indicators['prestigious_companies']:
            if company in text_lower:
                privilege_score += 1
        
        if privilege_score >= 3:
            return 'high'
        elif privilege_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def infer_diversity_indicators(self, text):
        """Infer diversity and inclusion indicators"""
        text_lower = text.lower()
        
        diversity_keywords = [
            'diversity', 'inclusion', 'equity', 'multicultural', 'inclusive',
            'affirmative action', 'equal opportunity', 'women in tech',
            'underrepresented', 'minority', 'lgbtq', 'accessibility'
        ]
        
        diversity_count = sum(1 for keyword in diversity_keywords if keyword in text_lower)
        
        if diversity_count >= 3:
            return 'high_diversity_focus'
        elif diversity_count >= 1:
            return 'some_diversity_focus'
        else:
            return 'no_diversity_focus'


class EnhancedFairnessMetrics:
    """Compute comprehensive fairness metrics for multi-class classification"""
    
    @staticmethod
    def demographic_parity_difference(y_pred, protected_attr):
        """Calculate demographic parity difference for multi-class"""
        protected_attr = np.array(protected_attr)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        group_positive_rates = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                # For multi-class, use overall positive rate
                positive_rate = np.mean(y_pred[group_mask] != -1)  # All predictions are "positive" in some class
                group_positive_rates.append(positive_rate)
        
        return max(group_positive_rates) - min(group_positive_rates) if group_positive_rates else 0
    
    @staticmethod
    def equal_opportunity_difference(y_true, y_pred, protected_attr, favorable_class=1):
        """Calculate equal opportunity difference"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        groups = np.unique(protected_attr)
        recall_rates = []
        
        for group in groups:
            group_mask = protected_attr == group
            true_positive_mask = group_mask & (y_true == favorable_class)
            if np.sum(true_positive_mask) > 0:
                recall = np.sum(y_pred[true_positive_mask] == favorable_class) / np.sum(true_positive_mask)
                recall_rates.append(recall)
        
        return max(recall_rates) - min(recall_rates) if recall_rates else 0
    
    @staticmethod
    def accuracy_equality_difference(y_true, y_pred, protected_attr):
        """Calculate accuracy equality difference"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        groups = np.unique(protected_attr)
        accuracies = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
                accuracies.append(accuracy)
        
        return max(accuracies) - min(accuracies) if accuracies else 0
    
    @staticmethod
    def statistical_parity_difference(y_pred, protected_attr, favorable_class=1):
        """Calculate statistical parity difference"""
        protected_attr = np.array(protected_attr)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        positive_rates = []
        
        for group in groups:
            group_mask = protected_attr == group
            if np.sum(group_mask) > 0:
                positive_rate = np.mean(y_pred[group_mask] == favorable_class)
                positive_rates.append(positive_rate)
        
        return max(positive_rates) - min(positive_rates) if positive_rates else 0


class EnhancedNameSubstitutionExperiment:
    """Enhanced name substitution experiments for bias measurement"""
    
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
        # Expanded name lists
        self.male_names = ['James', 'Robert', 'John', 'Michael', 'David', 'William', 
                          'Richard', 'Joseph', 'Thomas', 'Christopher', 'Daniel', 'Matthew']
        self.female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 
                            'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa']
        
        # Racially associated names (based on common associations)
        self.white_names = ['Emily', 'Anne', 'Jill', 'Allison', 'Laurie', 'Neil', 
                           'Geoffrey', 'Brett', 'Greg', 'Matthew']
        self.black_names = ['Lakisha', 'Latoya', 'Tamika', 'Imani', 'Ebony', 'Darnell',
                           'Jermaine', 'Tyrone', 'DeShawn', 'Marquis']
    
    def run_comprehensive_bias_experiment(self, test_texts, test_labels, num_samples=50):
        """Run comprehensive bias experiments with name substitution"""
        print("Running Comprehensive Bias Experiment...")
        
        results = {
            'gender_bias': self._run_gender_bias_experiment(test_texts, test_labels, num_samples),
            'racial_bias': self._run_racial_bias_experiment(test_texts, test_labels, num_samples)
        }
        
        return results
    
    def _run_gender_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure gender bias through name substitution"""
        male_predictions = []
        female_predictions = []
        bias_scores = []
        
        for i, (text, label) in enumerate(zip(test_texts[:num_samples], test_labels[:num_samples])):
            try:
                # Test with male name
                male_text = f"Candidate: {np.random.choice(self.male_names)} {text}"
                male_pred = self._predict_single(male_text)
                
                # Test with female name
                female_text = f"Candidate: {np.random.choice(self.female_names)} {text}"
                female_pred = self._predict_single(female_text)
                
                male_predictions.append(male_pred)
                female_predictions.append(female_pred)
                
                # Calculate bias score
                bias_score = abs(male_pred - female_pred)
                bias_scores.append(bias_score)
                
            except Exception as e:
                print(f"Error in gender bias experiment: {e}")
                continue
        
        return {
            'male_predictions': male_predictions,
            'female_predictions': female_predictions,
            'bias_scores': bias_scores,
            'average_bias': np.mean(bias_scores) if bias_scores else 0,
            'male_female_disparity': np.mean(male_predictions) - np.mean(female_predictions) if male_predictions and female_predictions else 0
        }
    
    def _run_racial_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure racial bias through name substitution"""
        white_predictions = []
        black_predictions = []
        bias_scores = []
        
        for i, (text, label) in enumerate(zip(test_texts[:num_samples], test_labels[:num_samples])):
            try:
                # Test with white-associated name
                white_text = f"Candidate: {np.random.choice(self.white_names)} {text}"
                white_pred = self._predict_single(white_text)
                
                # Test with black-associated name
                black_text = f"Candidate: {np.random.choice(self.black_names)} {text}"
                black_pred = self._predict_single(black_text)
                
                white_predictions.append(white_pred)
                black_predictions.append(black_pred)
                
                # Calculate bias score
                bias_score = abs(white_pred - black_pred)
                bias_scores.append(bias_score)
                
            except Exception as e:
                print(f"Error in racial bias experiment: {e}")
                continue
        
        return {
            'white_predictions': white_predictions,
            'black_predictions': black_predictions,
            'bias_scores': bias_scores,
            'average_bias': np.mean(bias_scores) if bias_scores else 0,
            'white_black_disparity': np.mean(white_predictions) - np.mean(black_predictions) if white_predictions and black_predictions else 0
        }
    
    def _predict_single(self, text):
        """Make prediction for single text"""
        try:
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
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5


class EnhancedBiasAnalyzer:
    """Comprehensive enhanced bias analysis framework"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.demographic_inference = EnhancedDemographicInference()
        self.fairness_metrics = EnhancedFairnessMetrics()
        self.name_experiment = EnhancedNameSubstitutionExperiment(tokenizer, model, device)
        
    def comprehensive_bias_analysis(self, test_texts, test_labels, test_predictions):
        """Run comprehensive enhanced bias analysis"""
        print("ENHANCED COMPREHENSIVE BIAS ANALYSIS")
        
        # Convert to numpy arrays
        test_labels = np.array(test_labels)
        test_predictions = np.array(test_predictions)
        
        # Enhanced demographic inference
        print("Enhanced demographic inference...")
        demographics = self._enhanced_infer_demographics(test_texts)
        
        # Calculate comprehensive fairness metrics
        print("Calculating enhanced fairness metrics...")
        fairness_results = self._calculate_enhanced_fairness_metrics(test_labels, test_predictions, demographics)
        
        # Enhanced category-level bias analysis
        print("Enhanced category-level bias analysis...")
        category_bias = self._enhanced_analyze_category_bias(test_labels, test_predictions, demographics)
        
        # Comprehensive name substitution experiments
        print("Running comprehensive name substitution experiments...")
        try:
            name_bias_results = self.name_experiment.run_comprehensive_bias_experiment(test_texts, test_labels, num_samples=50)
        except Exception as e:
            print(f"Name substitution experiment failed: {e}")
            name_bias_results = {
                'gender_bias': {'average_bias': 0.0, 'male_female_disparity': 0.0},
                'racial_bias': {'average_bias': 0.0, 'white_black_disparity': 0.0}
            }
        
        # Generate enhanced comprehensive report
        report = self._generate_enhanced_bias_report(fairness_results, category_bias, name_bias_results, test_labels, test_predictions)
        
        return report
    
    def _enhanced_infer_demographics(self, texts):
        """Enhanced demographic inference"""
        demographics = {
            'gender': [],
            'educational_privilege': [],
            'diversity_focus': []
        }
        
        for text in texts:
            demographics['gender'].append(self.demographic_inference.infer_gender(text))
            demographics['educational_privilege'].append(
                self.demographic_inference.infer_educational_privilege(text)
            )
            demographics['diversity_focus'].append(
                self.demographic_inference.infer_diversity_indicators(text)
            )
        
        # Print demographic distribution
        print("\nDemographic Distribution:")
        for demo_type, values in demographics.items():
            dist = Counter(values)
            print(f"  {demo_type}: {dict(dist)}")
        
        return demographics
    
    def _calculate_enhanced_fairness_metrics(self, y_true, y_pred, demographics):
        """Calculate enhanced fairness metrics"""
        metrics = {}
        
        for demo_type, demo_values in demographics.items():
            print(f"  Calculating {demo_type} fairness...")
            
            # Convert to numerical for metric calculation
            demo_encoder = LabelEncoder()
            demo_encoded = demo_encoder.fit_transform(demo_values)
            
            try:
                metrics[demo_type] = {
                    'demographic_parity': self.fairness_metrics.demographic_parity_difference(y_pred, demo_encoded),
                    'equal_opportunity': self.fairness_metrics.equal_opportunity_difference(y_true, y_pred, demo_encoded),
                    'accuracy_equality': self.fairness_metrics.accuracy_equality_difference(y_true, y_pred, demo_encoded),
                    'statistical_parity': self.fairness_metrics.statistical_parity_difference(y_pred, demo_encoded)
                }
            except Exception as e:
                print(f"Error calculating {demo_type} metrics: {e}")
                metrics[demo_type] = {
                    'demographic_parity': 0.0,
                    'equal_opportunity': 0.0,
                    'accuracy_equality': 0.0,
                    'statistical_parity': 0.0
                }
        
        return metrics
    
    def _enhanced_analyze_category_bias(self, y_true, y_pred, demographics):
        """Enhanced analysis of bias across job categories"""
        category_bias = {}
        
        for category in range(len(self.label_map)):
            category_mask = y_true == category
            if np.sum(category_mask) > 5:  # Only analyze categories with sufficient samples
                try:
                    category_accuracy = accuracy_score(y_true[category_mask], y_pred[category_mask])
                    
                    # Enhanced demographic analysis
                    demo_analysis = {}
                    for demo_type, demo_values in demographics.items():
                        demo_encoder = LabelEncoder()
                        demo_encoded = demo_encoder.fit_transform(demo_values)
                        groups = demo_encoder.classes_
                        
                        group_accuracies = {}
                        group_counts = {}
                        for group in groups:
                            group_mask = np.array(demo_values) == group
                            combined_mask = category_mask & group_mask
                            if np.sum(combined_mask) > 2:  # Minimum samples per group
                                try:
                                    group_accuracy = accuracy_score(
                                        y_true[combined_mask], 
                                        y_pred[combined_mask]
                                    )
                                    group_accuracies[str(group)] = group_accuracy
                                    group_counts[str(group)] = np.sum(combined_mask)
                                except:
                                    group_accuracies[str(group)] = 0.0
                                    group_counts[str(group)] = np.sum(combined_mask)
                        
                        demo_analysis[demo_type] = {
                            'accuracies': group_accuracies,
                            'counts': group_counts
                        }
                    
                    category_name = self.label_map.get(str(category), f"Category_{category}")
                    
                    category_bias[category_name] = {
                        'overall_accuracy': category_accuracy,
                        'sample_count': np.sum(category_mask),
                        'demographic_analysis': demo_analysis
                    }
                except Exception as e:
                    print(f"Error analyzing category {category}: {str(e)}")
                    continue
        
        return category_bias
    
    def _generate_enhanced_bias_report(self, fairness_results, category_bias, name_bias_results, y_true, y_pred):
        """Generate enhanced comprehensive bias analysis report"""
        
        # Enhanced bias scores
        gender_bias_data = name_bias_results['gender_bias']
        racial_bias_data = name_bias_results['racial_bias']
        
        report = {
            'fairness_metrics': fairness_results,
            'category_bias_analysis': category_bias,
            'name_substitution_bias': {
                'gender_bias': gender_bias_data,
                'racial_bias': racial_bias_data
            },
            'overall_performance': {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
            },
            'recommendations': self._generate_enhanced_recommendations(fairness_results, category_bias, name_bias_results)
        }
        
        # Enhanced summary
        self._print_enhanced_bias_summary(report)
        
        return report
    
    def _generate_enhanced_recommendations(self, fairness_results, category_bias, name_bias_results):
        """Generate enhanced actionable recommendations"""
        recommendations = []
        
        # Analyze fairness metrics
        high_bias_threshold = 0.1
        medium_bias_threshold = 0.05
        
        for demo_type, metrics in fairness_results.items():
            if metrics['demographic_parity'] > high_bias_threshold:
                recommendations.append(
                    f"HIGH PRIORITY: Significant demographic parity disparity ({metrics['demographic_parity']:.3f}) for {demo_type}. "
                    f"Implement immediate preprocessing and balancing strategies."
                )
            elif metrics['demographic_parity'] > medium_bias_threshold:
                recommendations.append(
                    f"MEDIUM PRIORITY: Moderate demographic parity disparity ({metrics['demographic_parity']:.3f}) for {demo_type}. "
                    f"Consider dataset balancing and fairness constraints."
                )
            
            if metrics['equal_opportunity'] > high_bias_threshold:
                recommendations.append(
                    f"HIGH PRIORITY: Equal opportunity concerns ({metrics['equal_opportunity']:.3f}) for {demo_type}. "
                    f"Implement adversarial debiasing and threshold optimization."
                )
        
        # Name bias recommendations
        gender_bias = name_bias_results['gender_bias']['average_bias']
        racial_bias = name_bias_results['racial_bias']['average_bias']
        
        if gender_bias > 0.05:
            recommendations.append(
                f"NAME BIAS: Significant gender bias detected ({gender_bias:.3f}). "
                f"Remove names during preprocessing and implement blind evaluation."
            )
        
        if racial_bias > 0.05:
            recommendations.append(
                f"RACIAL BIAS: Significant racial bias detected ({racial_bias:.3f}). "
                f"Implement comprehensive debiasing pipeline and regular audits."
            )
        
        # Category-specific recommendations
        problematic_categories = []
        for category, data in category_bias.items():
            demo_analysis = data.get('demographic_analysis', {})
            for demo_type, analysis in demo_analysis.items():
                accuracies = analysis.get('accuracies', {})
                if len(accuracies) > 1:
                    max_diff = max(accuracies.values()) - min(accuracies.values())
                    if max_diff > 0.15:
                        problematic_categories.append((category, demo_type, max_diff))
        
        if problematic_categories:
            worst_category = max(problematic_categories, key=lambda x: x[2])
            recommendations.append(
                f"CATEGORY BIAS: High bias in {worst_category[0]} for {worst_category[1]} "
                f"(disparity: {worst_category[2]:.3f}). Focus debiasing efforts on this category."
            )
        
        # Add general recommendations if no specific issues found
        if not recommendations:
            recommendations = [
                "System shows generally fair performance. Continue monitoring.",
                "Implement regular bias audits with updated metrics.",
                "Consider proactive debiasing for continuous improvement."
            ]
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _print_enhanced_bias_summary(self, report):
        """Print enhanced comprehensive bias analysis summary"""
        print("\n" + "=" * 60)
        print("ENHANCED BIAS ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Overall performance
        perf = report['overall_performance']
        print(f"Overall Performance:")
        print(f"  Accuracy: {perf['accuracy']:.3f}")
        print(f"  F1 Score: {perf['f1']:.3f}")
        
        # Fairness metrics
        print(f"\nFAIRNESS METRICS:")
        for demo_type, metrics in report['fairness_metrics'].items():
            print(f"  {demo_type.upper():20s} | "
                  f"DemPar: {metrics['demographic_parity']:.3f} | "
                  f"EqOpp: {metrics['equal_opportunity']:.3f} | "
                  f"AccEq: {metrics['accuracy_equality']:.3f} | "
                  f"StatPar: {metrics['statistical_parity']:.3f}")
        
        # Name bias
        name_bias = report['name_substitution_bias']
        print(f"\nNAME-BASED BIAS:")
        print(f"  Gender Bias: {name_bias['gender_bias']['average_bias']:.3f}")
        print(f"  Gender Disparity: {name_bias['gender_bias']['male_female_disparity']:.3f}")
        print(f"  Racial Bias: {name_bias['racial_bias']['average_bias']:.3f}")
        print(f"  Racial Disparity: {name_bias['racial_bias']['white_black_disparity']:.3f}")
        
        # Recommendations
        print(f"\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        print("=" * 60)


class EnhancedBiasVisualization:
    """Generate enhanced visualizations for bias analysis"""
    
    @staticmethod
    def plot_comprehensive_fairness_metrics(fairness_results, save_path=None):
        """Plot comprehensive fairness metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            metrics_to_plot = ['demographic_parity', 'equal_opportunity', 'accuracy_equality', 'statistical_parity']
            titles = ['Demographic Parity', 'Equal Opportunity', 'Accuracy Equality', 'Statistical Parity']
            
            for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
                demo_types = list(fairness_results.keys())
                values = [fairness_results[demo][metric] for demo in demo_types]
                
                colors = ['red' if abs(val) > 0.1 else 'orange' if abs(val) > 0.05 else 'green' for val in values]
                
                bars = axes[idx].bar(demo_types, values, color=colors, alpha=0.7)
                axes[idx].set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
                axes[idx].set_ylabel('Disparity Score', fontsize=12)
                axes[idx].tick_params(axis='x', rotation=45)
                
                # Add threshold lines and annotations
                axes[idx].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High Bias Threshold')
                axes[idx].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Medium Bias Threshold')
                axes[idx].legend()
                
                # Add value annotations on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return True
        except Exception as e:
            print(f"Visualization error: {e}")
            return False
    
    @staticmethod
    def plot_category_performance_bias(category_bias, save_path=None):
        """Plot category performance with bias indicators"""
        try:
            categories = list(category_bias.keys())
            accuracies = [cat_data['overall_accuracy'] for cat_data in category_bias.values()]
            sample_counts = [cat_data['sample_count'] for cat_data in category_bias.values()]
            
            # Calculate bias scores for each category
            bias_scores = []
            for cat_data in category_bias.values():
                max_bias = 0
                demo_analysis = cat_data.get('demographic_analysis', {})
                for demo_type, analysis in demo_analysis.items():
                    accuracies_dict = analysis.get('accuracies', {})
                    if len(accuracies_dict) > 1:
                        bias = max(accuracies_dict.values()) - min(accuracies_dict.values())
                        max_bias = max(max_bias, bias)
                bias_scores.append(max_bias)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Accuracy vs Sample Count with bias coloring
            scatter = ax1.scatter(sample_counts, accuracies, c=bias_scores, cmap='Reds', s=100, alpha=0.7)
            ax1.set_xlabel('Sample Count')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Category Performance: Accuracy vs Sample Count (Color = Bias Score)')
            plt.colorbar(scatter, ax=ax1, label='Bias Score')
            
            # Add category labels for extreme points
            for i, category in enumerate(categories):
                if accuracies[i] < 0.7 or bias_scores[i] > 0.1:
                    ax1.annotate(category, (sample_counts[i], accuracies[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            
            # Plot 2: Accuracy distribution with bias indicators
            bars = ax2.bar(categories, accuracies, color=['red' if bias > 0.1 else 'orange' if bias > 0.05 else 'green' 
                                                         for bias in bias_scores], alpha=0.7)
            ax2.set_title('Category Accuracy with Bias Indicators')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticklabels(categories, rotation=45, ha='right')
            
            # Add bias score annotations
            for bar, bias in zip(bars, bias_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{bias:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return True
        except Exception as e:
            print(f"Visualization error: {e}")
            return False
