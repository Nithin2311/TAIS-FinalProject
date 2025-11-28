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
import itertools
import shap
import lime
import lime.lime_text
from lime import submodular_pick
import warnings
warnings.filterwarnings('ignore')


class LimeExplainer:
    """Enhanced LIME explainer for model predictions with better error handling"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=list(label_map.values()),
            verbose=False,
            random_state=42,
            bow=False,
            split_expression=lambda x: x.split()
        )
    
    def explain_prediction(self, text, num_features=10):
        """Generate LIME explanation for a prediction with robust error handling"""
        if not text or len(text.strip()) < 10:
            return None
            
        def predict_proba(texts):
            probabilities = []
            for text in texts:
                try:
                    inputs = self.tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=256,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        probabilities.append(probs.cpu().numpy()[0])
                except Exception as e:
                    probabilities.append(np.ones(len(self.label_map)) / len(self.label_map))
            
            return np.array(probabilities)
        
        try:
            text_str = str(text)
            if len(text_str.split()) < 3:
                return None
                
            exp = self.explainer.explain_instance(
                text_str,
                predict_proba,
                num_features=min(num_features, len(text_str.split())),
                num_samples=100,
                top_labels=1
            )
            return exp
        except Exception as e:
            print(f"LIME explanation generation failed: {e}")
            return None
    
    def plot_explanation(self, exp, label):
        """Plot LIME explanation as matplotlib figure with enhanced visualization"""
        try:
            if exp is None:
                return self._create_error_plot("No explanation generated")
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            exp_list = exp.as_list(label=label)
            
            if not exp_list:
                return self._create_error_plot("No features to explain")
            
            exp_list.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = exp_list[:10]
            
            features = [x[0] for x in top_features]
            weights = [x[1] for x in top_features]
            
            colors = ['green' if w > 0 else 'red' for w in weights]
            y_pos = np.arange(len(features))
            
            ax.barh(y_pos, weights, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'LIME Explanation for {self.label_map.get(str(label), f"Class {label}")}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            for i, v in enumerate(weights):
                ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', 
                       color='black', fontsize=9, va='center',
                       ha='left' if v >= 0 else 'right')
            
            plt.tight_layout()
            
            buf = plt.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf
            
        except Exception as e:
            print(f"LIME plot generation failed: {e}")
            return self._create_error_plot(f"Explanation error: {str(e)}")
    
    def _create_error_plot(self, message):
        """Create an error message plot"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, message, 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, wrap=True)
        ax.axis('off')
        
        buf = plt.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf


class EnhancedDemographicInference:
    """Enhanced demographic inference from resume text"""
    
    def __init__(self):
        self.gender_patterns = {
            'male': [
                r'\bhe\b', r'\bhim\b', r'\bhis\b', r'\bmale\b', r'\bman\b', r'\bmen\b',
                r'\bboy\b', r'\bmr\.', r'\bmr\b', r'\bmister\b', r'\bfather\b', r'\bhusband\b',
                r'\bbrother\b', r'\bson\b', r'\bgentleman\b'
            ],
            'female': [
                r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bfemale\b', r'\bwoman\b', r'\bwomen\b',
                r'\bgirl\b', r'\bms\.', r'\bms\b', r'\bmiss\b', r'\bmrs\.', r'\bmrs\b',
                r'\bmother\b', r'\bwife\b', r'\bsister\b', r'\bdaughter\b', r'\blady\b'
            ]
        }
        
        self.privilege_indicators = {
            'elite_universities': [
                'harvard', 'stanford', 'mit', 'princeton', 'yale', 'columbia',
                'cambridge', 'oxford', 'caltech', 'cornell', 'upenn', 'duke',
                'johns hopkins', 'northwestern', 'brown', 'dartmouth'
            ],
            'prestigious_companies': [
                'google', 'microsoft', 'apple', 'amazon', 'meta', 'goldman sachs',
                'mckinsey', 'bain', 'boston consulting', 'jpmorgan', 'morgan stanley',
                'goldman', 'blackrock', 'bridgewater'
            ]
        }
        
        self.age_indicators = {
            'experience_years': [r'(\d+)\s+years', r'(\d+)\+ years', r'(\d+)-(\d+) years'],
            'graduation_years': [r'class of (\d{4})', r'graduated (\d{4})', r'\b(\d{4})\s*-\s*(\d{4})'],
            'age_phrases': ['recent graduate', 'entry level', 'junior', 'senior', 'experienced', 'veteran']
        }
        
        self.disability_indicators = [
            'disability', 'disabled', 'handicap', 'special needs', 'accessibility',
            'accommodation', 'able-bodied', 'differently abled', 'inclusion',
            'ada', 'americans with disabilities act'
        ]
    
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
            'underrepresented', 'minority', 'lgbtq', 'accessibility',
            'belonging', 'equality', 'social justice', 'inclusive workplace'
        ]
        
        diversity_count = sum(1 for keyword in diversity_keywords if keyword in text_lower)
        
        if diversity_count >= 3:
            return 'high_diversity_focus'
        elif diversity_count >= 1:
            return 'some_diversity_focus'
        else:
            return 'no_diversity_focus'
    
    def infer_age_group(self, text):
        """Infer approximate age group from text patterns"""
        text_lower = text.lower()
        
        experience_match = re.search(r'(\d+)\s+years', text_lower)
        if experience_match:
            years_exp = int(experience_match.group(1))
            if years_exp <= 3:
                return 'early_career'
            elif years_exp <= 10:
                return 'mid_career'
            else:
                return 'senior'
        
        graduation_match = re.search(r'(?:class of|graduated)\s+(\d{4})', text_lower)
        if graduation_match:
            grad_year = int(graduation_match.group(1))
            estimated_age = 2024 - grad_year + 22
            if estimated_age < 30:
                return 'young'
            elif estimated_age < 50:
                return 'middle_aged'
            else:
                return 'senior'
        
        if any(phrase in text_lower for phrase in ['recent graduate', 'entry level', 'new grad']):
            return 'early_career'
        elif any(phrase in text_lower for phrase in ['senior', 'veteran', 'experienced']):
            return 'senior'
        
        return 'unknown'
    
    def infer_disability_status(self, text):
        """Infer potential disability status mentions"""
        text_lower = text.lower()
        
        disability_count = sum(1 for indicator in self.disability_indicators if indicator in text_lower)
        
        if disability_count >= 2:
            return 'disability_mentioned'
        elif disability_count == 1:
            return 'possibly_disability'
        else:
            return 'no_mention'


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
                positive_rate = np.mean(y_pred[group_mask] != -1)
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
    
    @staticmethod
    def disparate_impact_ratio(y_pred, protected_attr, favorable_class=1, reference_group=None):
        """Calculate disparate impact ratio (80% rule)"""
        protected_attr = np.array(protected_attr)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        
        if reference_group is None:
            reference_group = max(groups, key=lambda x: np.sum(protected_attr == x))
        
        reference_mask = protected_attr == reference_group
        reference_rate = np.mean(y_pred[reference_mask] == favorable_class) if np.sum(reference_mask) > 0 else 0
        
        impact_ratios = {}
        for group in groups:
            if group != reference_group:
                group_mask = protected_attr == group
                if np.sum(group_mask) > 0:
                    group_rate = np.mean(y_pred[group_mask] == favorable_class)
                    impact_ratio = group_rate / reference_rate if reference_rate > 0 else 0
                    impact_ratios[group] = impact_ratio
        
        return impact_ratios
    
    @staticmethod
    def intersectional_fairness_analysis(y_true, y_pred, protected_attrs):
        """Analyze fairness across intersectional groups"""
        intersectional_groups = {}
        
        unique_combinations = set(zip(*protected_attrs.values()))
        
        for combo in unique_combinations:
            group_mask = np.ones(len(y_true), dtype=bool)
            group_name_parts = []
            
            for i, (attr_name, attr_values) in enumerate(protected_attrs.items()):
                group_mask &= (attr_values == combo[i])
                group_name_parts.append(f"{attr_name}_{combo[i]}")
            
            group_name = "_".join(group_name_parts)
            
            if np.sum(group_mask) > 0:
                group_accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
                intersectional_groups[group_name] = {
                    'accuracy': group_accuracy,
                    'size': np.sum(group_mask),
                    'combination': combo
                }
        
        return intersectional_groups


class EnhancedNameSubstitutionExperiment:
    """Enhanced name substitution experiments for bias measurement"""
    
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
        self.male_names = ['James', 'Robert', 'John', 'Michael', 'David', 'William', 
                          'Richard', 'Joseph', 'Thomas', 'Christopher', 'Daniel', 'Matthew',
                          'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua']
        self.female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 
                            'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa',
                            'Betty', 'Margaret', 'Sandra', 'Ashley', 'Dorothy', 'Kimberly']
        
        self.white_names = ['Emily', 'Anne', 'Jill', 'Allison', 'Laurie', 'Neil', 
                           'Geoffrey', 'Brett', 'Greg', 'Matthew', 'Katie', 'Megan']
        self.black_names = ['Lakisha', 'Latoya', 'Tamika', 'Imani', 'Ebony', 'Darnell',
                           'Jermaine', 'Tyrone', 'DeShawn', 'Marquis', 'Shanice', 'Aaliyah']
        self.asian_names = ['Wei', 'Jing', 'Li', 'Zhang', 'Wang', 'Chen', 
                           'Yong', 'Min', 'Hui', 'Xiao', 'Mei', 'Lin']
        self.hispanic_names = ['Jose', 'Carlos', 'Luis', 'Juan', 'Miguel', 'Rosa',
                              'Maria', 'Carmen', 'Ana', 'Dolores', 'Sofia', 'Isabella']
    
    def run_comprehensive_bias_experiment(self, test_texts, test_labels, num_samples=50):
        """Run comprehensive bias experiments with name substitution"""
        print("Running Enhanced Comprehensive Bias Experiment...")
        
        results = {
            'gender_bias': self._run_gender_bias_experiment(test_texts, test_labels, num_samples),
            'racial_bias': self._run_racial_bias_experiment(test_texts, test_labels, num_samples),
            'intersectional_bias': self._run_intersectional_bias_experiment(test_texts, test_labels, num_samples)
        }
        
        return results
    
    def _run_gender_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure gender bias through name substitution"""
        male_predictions = []
        female_predictions = []
        bias_scores = []
        confidence_changes = []
        
        for i, (text, label) in enumerate(zip(test_texts[:num_samples], test_labels[:num_samples])):
            try:
                male_text = f"Candidate: {np.random.choice(self.male_names)} {text}"
                male_pred, male_conf = self._predict_single(male_text)
                
                female_text = f"Candidate: {np.random.choice(self.female_names)} {text}"
                female_pred, female_conf = self._predict_single(female_text)
                
                male_predictions.append(male_pred)
                female_predictions.append(female_pred)
                
                bias_score = 1 if male_pred != female_pred else 0
                bias_scores.append(bias_score)
                
                conf_change = abs(male_conf - female_conf)
                confidence_changes.append(conf_change)
                
            except Exception as e:
                print(f"Error in gender bias experiment: {e}")
                continue
        
        return {
            'male_predictions': male_predictions,
            'female_predictions': female_predictions,
            'bias_scores': bias_scores,
            'average_bias': np.mean(bias_scores) if bias_scores else 0,
            'male_female_disparity': np.mean(male_predictions) - np.mean(female_predictions) if male_predictions and female_predictions else 0,
            'confidence_disparity': np.mean(confidence_changes) if confidence_changes else 0
        }
    
    def _run_racial_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure racial bias through name substitution"""
        racial_groups = {
            'white': self.white_names,
            'black': self.black_names, 
            'asian': self.asian_names,
            'hispanic': self.hispanic_names
        }
        
        group_predictions = {group: [] for group in racial_groups.keys()}
        bias_matrix = np.zeros((len(racial_groups), len(racial_groups)))
        
        for i, (text, label) in enumerate(zip(test_texts[:num_samples], test_labels[:num_samples])):
            try:
                predictions = {}
                
                for group, names in racial_groups.items():
                    group_text = f"Candidate: {np.random.choice(names)} {text}"
                    pred, conf = self._predict_single(group_text)
                    predictions[group] = pred
                    group_predictions[group].append(pred)
                
                groups = list(racial_groups.keys())
                for j, group1 in enumerate(groups):
                    for k, group2 in enumerate(groups):
                        if j != k and predictions[group1] != predictions[group2]:
                            bias_matrix[j, k] += 1
                
            except Exception as e:
                print(f"Error in racial bias experiment: {e}")
                continue
        
        total_comparisons = num_samples * (len(racial_groups) * (len(racial_groups) - 1))
        average_bias = np.sum(bias_matrix) / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'group_predictions': {k: (np.mean(v) if v else 0) for k, v in group_predictions.items()},
            'bias_matrix': bias_matrix.tolist(),
            'average_bias': average_bias,
            'white_black_disparity': np.mean(group_predictions['white']) - np.mean(group_predictions['black']) if group_predictions['white'] and group_predictions['black'] else 0
        }
    
    def _run_intersectional_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure intersectional bias (gender + race)"""
        intersectional_results = {}
        
        intersectional_groups = [
            ('white', 'male'), ('white', 'female'),
            ('black', 'male'), ('black', 'female'), 
            ('asian', 'male'), ('asian', 'female'),
            ('hispanic', 'male'), ('hispanic', 'female')
        ]
        
        for race, gender in intersectional_groups:
            race_names = getattr(self, f"{race}_names")
            gender_names = getattr(self, f"{gender}_names")
            
            if race_names and gender_names:
                group_key = f"{race}_{gender}"
                predictions = []
                
                for i, (text, label) in enumerate(zip(test_texts[:20], test_labels[:20])):
                    try:
                        name = np.random.choice([n for n in gender_names if n in race_names or True])
                        group_text = f"Candidate: {name} {text}"
                        pred, conf = self._predict_single(group_text)
                        predictions.append(pred)
                    except:
                        continue
                
                if predictions:
                    intersectional_results[group_key] = {
                        'mean_prediction': np.mean(predictions),
                        'std_prediction': np.std(predictions),
                        'sample_size': len(predictions)
                    }
        
        return intersectional_results
    
    def _predict_single(self, text):
        """Make prediction for single text with confidence"""
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
                max_prob, predicted_class = torch.max(probs, dim=1)
                return predicted_class.item(), max_prob.item()
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.5


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
        
        test_labels = np.array(test_labels)
        test_predictions = np.array(test_predictions)
        
        print("Enhanced demographic inference...")
        demographics = self._enhanced_infer_demographics(test_texts)
        
        print("Calculating enhanced fairness metrics...")
        fairness_results = self._calculate_enhanced_fairness_metrics(test_labels, test_predictions, demographics)
        
        print("Enhanced category-level bias analysis...")
        category_bias = self._enhanced_analyze_category_bias(test_labels, test_predictions, demographics)
        
        print("Intersectional bias analysis...")
        intersectional_results = self._analyze_intersectional_bias(test_labels, test_predictions, demographics)
        
        print("Running comprehensive name substitution experiments...")
        try:
            name_bias_results = self.name_experiment.run_comprehensive_bias_experiment(test_texts, test_labels, num_samples=50)
        except Exception as e:
            print(f"Name substitution experiment failed: {e}")
            name_bias_results = {
                'gender_bias': {'average_bias': 0.0, 'male_female_disparity': 0.0},
                'racial_bias': {'average_bias': 0.0, 'white_black_disparity': 0.0},
                'intersectional_bias': {}
            }
        
        print("Counterfactual fairness analysis...")
        counterfactual_results = self._analyze_counterfactual_fairness(test_texts, test_labels)
        
        report = self._generate_enhanced_bias_report(
            fairness_results, category_bias, name_bias_results, 
            intersectional_results, counterfactual_results,
            test_labels, test_predictions
        )
        
        return report
    
    def _enhanced_infer_demographics(self, texts):
        """Enhanced demographic inference with more attributes"""
        demographics = {
            'gender': [],
            'educational_privilege': [],
            'diversity_focus': [],
            'age_group': [],
            'disability_status': []
        }
        
        for text in texts:
            demographics['gender'].append(self.demographic_inference.infer_gender(text))
            demographics['educational_privilege'].append(
                self.demographic_inference.infer_educational_privilege(text)
            )
            demographics['diversity_focus'].append(
                self.demographic_inference.infer_diversity_indicators(text)
            )
            demographics['age_group'].append(
                self.demographic_inference.infer_age_group(text)
            )
            demographics['disability_status'].append(
                self.demographic_inference.infer_disability_status(text)
            )
        
        print("\nEnhanced Demographic Distribution:")
        for demo_type, values in demographics.items():
            dist = Counter(values)
            print(f"  {demo_type}: {dict(dist)}")
        
        return demographics
    
    def _calculate_enhanced_fairness_metrics(self, y_true, y_pred, demographics):
        """Calculate enhanced fairness metrics with more attributes"""
        metrics = {}
        
        for demo_type, demo_values in demographics.items():
            print(f"  Calculating {demo_type} fairness...")
            
            demo_encoder = LabelEncoder()
            demo_encoded = demo_encoder.fit_transform(demo_values)
            
            try:
                metrics[demo_type] = {
                    'demographic_parity': self.fairness_metrics.demographic_parity_difference(y_pred, demo_encoded),
                    'equal_opportunity': self.fairness_metrics.equal_opportunity_difference(y_true, y_pred, demo_encoded),
                    'accuracy_equality': self.fairness_metrics.accuracy_equality_difference(y_true, y_pred, demo_encoded),
                    'statistical_parity': self.fairness_metrics.statistical_parity_difference(y_pred, demo_encoded),
                    'disparate_impact': self.fairness_metrics.disparate_impact_ratio(y_pred, demo_values)
                }
            except Exception as e:
                print(f"Error calculating {demo_type} metrics: {e}")
                metrics[demo_type] = {
                    'demographic_parity': 0.0,
                    'equal_opportunity': 0.0,
                    'accuracy_equality': 0.0,
                    'statistical_parity': 0.0,
                    'disparate_impact': {}
                }
        
        return metrics
    
    def _enhanced_analyze_category_bias(self, y_true, y_pred, demographics):
        """Enhanced analysis of bias across job categories"""
        category_bias = {}
        
        for category in range(len(self.label_map)):
            category_mask = y_true == category
            if np.sum(category_mask) > 5:
                try:
                    category_accuracy = accuracy_score(y_true[category_mask], y_pred[category_mask])
                    
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
                            if np.sum(combined_mask) > 2:
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
                            'counts': group_counts,
                            'max_disparity': max(group_accuracies.values()) - min(group_accuracies.values()) if group_accuracies else 0
                        }
                    
                    category_name = self.label_map.get(str(category), f"Category_{category}")
                    
                    category_bias[category_name] = {
                        'overall_accuracy': category_accuracy,
                        'sample_count': np.sum(category_mask),
                        'demographic_analysis': demo_analysis,
                        'bias_score': self._calculate_category_bias_score(demo_analysis)
                    }
                except Exception as e:
                    print(f"Error analyzing category {category}: {str(e)}")
                    continue
        
        return category_bias
    
    def _analyze_intersectional_bias(self, y_true, y_pred, demographics):
        """Analyze bias across intersectional groups"""
        print("  Performing intersectional bias analysis...")
        
        protected_attrs = {
            'gender': demographics['gender'],
            'privilege': demographics['educational_privilege']
        }
        
        intersectional_results = self.fairness_metrics.intersectional_fairness_analysis(
            y_true, y_pred, protected_attrs
        )
        
        return intersectional_results
    
    def _analyze_counterfactual_fairness(self, test_texts, test_labels, num_samples=30):
        """Analyze counterfactual fairness by modifying demographic indicators"""
        print("  Performing counterfactual fairness analysis...")
        
        counterfactual_changes = []
        
        for i, (text, label) in enumerate(zip(test_texts[:num_samples], test_labels[:num_samples])):
            try:
                orig_pred, orig_conf = self._predict_single_text(text)
                
                gender_modified = self._modify_gender_indicators(text)
                cf_pred, cf_conf = self._predict_single_text(gender_modified)
                
                if orig_pred != cf_pred:
                    counterfactual_changes.append(1)
                else:
                    counterfactual_changes.append(0)
                    
            except Exception as e:
                print(f"Counterfactual analysis error: {e}")
                continue
        
        return {
            'counterfactual_unfairness_rate': np.mean(counterfactual_changes) if counterfactual_changes else 0,
            'total_counterfactuals': len(counterfactual_changes)
        }
    
    def _modify_gender_indicators(self, text):
        """Modify gender indicators in text for counterfactual analysis"""
        modified_text = text.lower()
        
        male_to_female = {
            ' he ': ' she ', ' him ': ' her ', ' his ': ' her ', ' male ': ' female ',
            ' man ': ' woman ', ' men ': ' women ', ' mr. ': ' ms. ', ' mister ': ' miss '
        }
        
        female_to_male = {v: k for k, v in male_to_female.items()}
        
        for male, female in male_to_female.items():
            if male in modified_text:
                modified_text = modified_text.replace(male, female)
            elif female in modified_text:
                modified_text = modified_text.replace(female, male)
        
        return modified_text
    
    def _predict_single_text(self, text):
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
                max_prob, predicted_class = torch.max(probs, dim=1)
                return predicted_class.item(), max_prob.item()
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.5
    
    def _calculate_category_bias_score(self, demo_analysis):
        """Calculate overall bias score for a category"""
        max_disparities = []
        
        for demo_type, analysis in demo_analysis.items():
            if 'max_disparity' in analysis:
                max_disparities.append(analysis['max_disparity'])
        
        return np.mean(max_disparities) if max_disparities else 0.0
    
    def _generate_enhanced_bias_report(self, fairness_results, category_bias, name_bias_results, 
                                     intersectional_results, counterfactual_results, y_true, y_pred):
        """Generate enhanced comprehensive bias analysis report"""
        
        report = {
            'fairness_metrics': fairness_results,
            'category_bias_analysis': category_bias,
            'name_substitution_bias': name_bias_results,
            'intersectional_bias_analysis': intersectional_results,
            'counterfactual_fairness': counterfactual_results,
            'overall_performance': {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
            },
            'bias_heatmap_data': self._generate_bias_heatmap_data(category_bias),
            'recommendations': self._generate_enhanced_recommendations(
                fairness_results, category_bias, name_bias_results, intersectional_results
            )
        }
        
        self._print_enhanced_bias_summary(report)
        
        return report
    
    def _generate_bias_heatmap_data(self, category_bias):
        """Generate data for bias heatmap visualization"""
        heatmap_data = {}
        
        for category, data in category_bias.items():
            bias_scores = {}
            demo_analysis = data.get('demographic_analysis', {})
            
            for demo_type, analysis in demo_analysis.items():
                bias_scores[demo_type] = analysis.get('max_disparity', 0)
            
            heatmap_data[category] = {
                'overall_accuracy': data.get('overall_accuracy', 0),
                'bias_scores': bias_scores,
                'bias_score': data.get('bias_score', 0)
            }
        
        return heatmap_data
    
    def _generate_enhanced_recommendations(self, fairness_results, category_bias, name_bias_results, intersectional_results):
        """Generate enhanced actionable recommendations"""
        recommendations = []
        
        high_bias_threshold = 0.1
        medium_bias_threshold = 0.05
        
        for demo_type, metrics in fairness_results.items():
            dp_score = metrics.get('demographic_parity', 0)
            eo_score = metrics.get('equal_opportunity', 0)
            
            if dp_score > high_bias_threshold:
                recommendations.append(
                    f"HIGH PRIORITY: Significant demographic parity disparity ({dp_score:.3f}) for {demo_type}. "
                    f"Implement immediate preprocessing and reweighting strategies."
                )
            elif dp_score > medium_bias_threshold:
                recommendations.append(
                    f"MEDIUM PRIORITY: Moderate demographic parity disparity ({dp_score:.3f}) for {demo_type}. "
                    f"Consider dataset balancing and fairness-aware training."
                )
            
            if eo_score > high_bias_threshold:
                recommendations.append(
                    f"HIGH PRIORITY: Equal opportunity concerns ({eo_score:.3f}) for {demo_type}. "
                    f"Implement adversarial debiasing and threshold optimization."
                )
        
        gender_bias = name_bias_results.get('gender_bias', {}).get('average_bias', 0)
        racial_bias = name_bias_results.get('racial_bias', {}).get('average_bias', 0)
        
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
        
        high_bias_categories = []
        for category, data in category_bias.items():
            bias_score = data.get('bias_score', 0)
            accuracy = data.get('overall_accuracy', 0)
            
            if bias_score > 0.15 and accuracy < 0.7:
                high_bias_categories.append((category, bias_score, accuracy))
        
        if high_bias_categories:
            worst_category = max(high_bias_categories, key=lambda x: x[1])
            recommendations.append(
                f"CATEGORY BIAS: High bias and low accuracy in {worst_category[0]} "
                f"(bias: {worst_category[1]:.3f}, accuracy: {worst_category[2]:.3f}). "
                f"Focus debiasing and data augmentation on this category."
            )
        
        if intersectional_results:
            worst_intersectional = min(intersectional_results.items(), 
                                     key=lambda x: x[1].get('accuracy', 1))
            if worst_intersectional[1].get('accuracy', 1) < 0.6:
                recommendations.append(
                    f"INTERSECTIONAL BIAS: Worst-performing group {worst_intersectional[0]} "
                    f"(accuracy: {worst_intersectional[1].get('accuracy', 0):.3f}). "
                    f"Implement intersectional debiasing strategies."
                )
        
        if not recommendations:
            recommendations = [
                "System shows generally fair performance. Continue monitoring.",
                "Implement regular bias audits with updated metrics.",
                "Consider proactive debiasing for continuous improvement."
            ]
        
        return recommendations[:8]
    
    def _print_enhanced_bias_summary(self, report):
        """Print enhanced comprehensive bias analysis summary"""
        print("\n" + "=" * 60)
        print("ENHANCED BIAS ANALYSIS SUMMARY")
        print("=" * 60)
        
        perf = report['overall_performance']
        print(f"Overall Performance:")
        print(f"  Accuracy: {perf['accuracy']:.3f}")
        print(f"  F1 Score: {perf['f1']:.3f}")
        
        print(f"\nFAIRNESS METRICS:")
        for demo_type, metrics in report['fairness_metrics'].items():
            print(f"  {demo_type.upper():20s} | "
                  f"DemPar: {metrics['demographic_parity']:.3f} | "
                  f"EqOpp: {metrics['equal_opportunity']:.3f} | "
                  f"AccEq: {metrics['accuracy_equality']:.3f}")
        
        name_bias = report['name_substitution_bias']
        print(f"\nNAME-BASED BIAS:")
        print(f"  Gender Bias: {name_bias['gender_bias']['average_bias']:.3f}")
        print(f"  Racial Bias: {name_bias['racial_bias']['average_bias']:.3f}")
        
        cf_fairness = report['counterfactual_fairness']
        print(f"  Counterfactual Unfairness: {cf_fairness['counterfactual_unfairness_rate']:.3f}")
        
        category_bias = report['category_bias_analysis']
        high_bias_categories = sorted(
            [(cat, data['bias_score']) for cat, data in category_bias.items() if data['bias_score'] > 0.1],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if high_bias_categories:
            print(f"\nHIGHEST BIAS CATEGORIES:")
            for cat, bias in high_bias_categories:
                print(f"  {cat}: {bias:.3f}")
        
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
                
                axes[idx].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High Bias Threshold')
                axes[idx].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Medium Bias Threshold')
                axes[idx].legend()
                
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
            
            scatter = ax1.scatter(sample_counts, accuracies, c=bias_scores, cmap='Reds', s=100, alpha=0.7)
            ax1.set_xlabel('Sample Count')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Category Performance: Accuracy vs Sample Count (Color = Bias Score)')
            plt.colorbar(scatter, ax=ax1, label='Bias Score')
            
            for i, category in enumerate(categories):
                if accuracies[i] < 0.7 or bias_scores[i] > 0.1:
                    ax1.annotate(category, (sample_counts[i], accuracies[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            
            bars = ax2.bar(categories, accuracies, color=['red' if bias > 0.1 else 'orange' if bias > 0.05 else 'green' 
                                                         for bias in bias_scores], alpha=0.7)
            ax2.set_title('Category Accuracy with Bias Indicators')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticklabels(categories, rotation=45, ha='right')
            
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
    
    @staticmethod
    def plot_intersectional_bias_heatmap(intersectional_results, save_path=None):
        """Plot intersectional bias as heatmap"""
        try:
            if not intersectional_results:
                return False
                
            groups = list(intersectional_results.keys())
            accuracies = [intersectional_results[group]['accuracy'] for group in groups]
            sizes = [intersectional_results[group]['size'] for group in groups]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            accuracy_matrix = np.array(accuracies).reshape(1, -1)
            im1 = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto')
            ax1.set_xticks(range(len(groups)))
            ax1.set_xticklabels(groups, rotation=45, ha='right')
            ax1.set_title('Intersectional Group Accuracies')
            plt.colorbar(im1, ax=ax1)
            
            for i, acc in enumerate(accuracies):
                ax1.text(i, 0, f'{acc:.2f}', ha='center', va='center', fontweight='bold')
            
            size_matrix = np.array(sizes).reshape(1, -1)
            im2 = ax2.imshow(size_matrix, cmap='Blues', aspect='auto')
            ax2.set_xticks(range(len(groups)))
            ax2.set_xticklabels(groups, rotation=45, ha='right')
            ax2.set_title('Intersectional Group Sizes')
            plt.colorbar(im2, ax=ax2)
            
            for i, size in enumerate(sizes):
                ax2.text(i, 0, str(size), ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return True
        except Exception as e:
            print(f"Heatmap visualization error: {e}")
            return False
