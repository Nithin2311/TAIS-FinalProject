"""
Bias detection and fairness analysis module.
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
import warnings
warnings.filterwarnings('ignore')


class LimeExplainer:
    """LIME explainer for model predictions"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        
        import lime.lime_text
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=list(label_map.values()),
            verbose=False,
            random_state=42,
            bow=False,
            split_expression=lambda x: x.split()
        )
    
    def explain_prediction(self, text, num_features=10):
        """Generate LIME explanation for a prediction"""
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
                        max_length=512,
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
            if len(text_str.split()) < 5:
                return None
                
            exp = self.explainer.explain_instance(
                text_str,
                predict_proba,
                num_features=min(num_features, len(text_str.split())),
                num_samples=200,
                top_labels=1
            )
            return exp
        except Exception as e:
            print(f"LIME explanation generation failed: {e}")
            return None


class DemographicInference:
    """Demographic inference from resume text"""
    
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
        
        self.age_indicators = {
            'experience_years': [r'(\d+)\s+years', r'(\d+)\+ years', r'(\d+)-(\d+) years'],
            'graduation_years': [r'class of (\d{4})', r'graduated (\d{4})', r'\b(\d{4})\s*-\s*(\d{4})'],
            'age_phrases': ['recent graduate', 'entry level', 'junior', 'senior', 'experienced', 'veteran']
        }
        
        self.race_name_patterns = {
            'black': ['lakisha', 'latoya', 'tamika', 'imani', 'ebony', 'darnell',
                     'jermaine', 'tyrone', 'deshawn', 'marquis', 'shanice', 'aaliyah'],
            'white': ['emily', 'anne', 'jill', 'allison', 'laurie', 'neil',
                     'geoffrey', 'brett', 'greg', 'matthew', 'katie', 'megan'],
            'asian': ['wei', 'jing', 'li', 'zhang', 'wang', 'chen',
                     'yong', 'min', 'hui', 'xiao', 'mei', 'lin'],
            'hispanic': ['jose', 'carlos', 'luis', 'juan', 'miguel', 'rosa',
                        'maria', 'carmen', 'ana', 'dolores', 'sofia', 'isabella']
        }
        
        # ADDED: Job-related keyword patterns for anonymized resumes
        self.job_gender_patterns = {
            'male_leaning': [
                'football', 'basketball', 'baseball', 'hockey', 'golf', 'fishing', 'hunting',
                'military', 'marine', 'navy', 'army', 'combat', 'veteran',
                'engineering', 'mechanical', 'electrical', 'construction', 'contractor',
                'programming', 'coding', 'software', 'algorithm', 'hackathon', 'devops',
                'physics', 'mathematics', 'calculus', 'quantum', 'statistics',
                'motorcycle', 'cars', 'automotive', 'welding', 'carpentry'
            ],
            'female_leaning': [
                'nursing', 'caregiving', 'childcare', 'preschool', 'kindergarten',
                'counseling', 'therapy', 'social work', 'human resources', 'hr',
                'public relations', 'event planning', 'interior design', 'decorating',
                'early childhood', 'family therapy', 'community outreach', 'volunteer',
                'teaching', 'education', 'curriculum', 'pedagogy', 'tutoring',
                'administrative', 'receptionist', 'secretarial', 'office manager',
                'fashion', 'beauty', 'cosmetology', 'hairstyling', 'makeup'
            ]
        }
    
    def infer_gender(self, text):
        """Infer gender from text using pronouns and job-related keywords"""
        text_lower = text.lower()
        
        male_score = 0
        female_score = 0
        
        # Check pronouns
        for pattern in self.gender_patterns['male']:
            male_score += len(re.findall(pattern, text_lower))
        
        for pattern in self.gender_patterns['female']:
            female_score += len(re.findall(pattern, text_lower))
        
        # Check job-related keywords (for anonymized resumes)
        for keyword in self.job_gender_patterns['male_leaning']:
            if keyword in text_lower:
                male_score += 1
        
        for keyword in self.job_gender_patterns['female_leaning']:
            if keyword in text_lower:
                female_score += 1
        
        if male_score > female_score:
            return 'male'
        elif female_score > male_score:
            return 'female'
        else:
            return 'unknown'
    
    def infer_race_from_names(self, text):
        """Infer race/ethnicity from names in text - improved version"""
        text_lower = text.lower()
        
        # Enhanced name patterns with more comprehensive lists
        self.race_name_patterns = {
            'black': [
                'darnell', 'lakisha', 'latoya', 'tamika', 'imani', 'ebony', 
                'jermaine', 'tyrone', 'deshawn', 'marquis', 'shanice', 'aaliyah',
                'kareem', 'latonya', 'tyrell', 'shaniqua', 'deandre', 'keisha',
                'jamal', 'tanisha', 'malik', 'tia', 'darius', 'lashonda'
            ],
            'white': [
                'emily', 'anne', 'jill', 'allison', 'laurie', 'neil',
                'geoffrey', 'brett', 'greg', 'matthew', 'katie', 'megan',
                'james', 'robert', 'john', 'michael', 'david', 'william',
                'richard', 'joseph', 'thomas', 'christopher', 'daniel',
                'mary', 'patricia', 'jennifer', 'linda', 'elizabeth'
            ],
            'asian': [
                'chen', 'wei', 'jing', 'li', 'zhang', 'wang', 
                'yong', 'min', 'hui', 'xiao', 'mei', 'lin',
                'kim', 'park', 'choi', 'lee', 'jung', 'kang',
                'tanaka', 'sato', 'suzuki', 'takahashi', 'watanabe',
                'patel', 'sharma', 'kumar', 'singh', 'gupta'
            ],
            'hispanic': [
                'garcia', 'rodriguez', 'martinez', 'hernandez', 'lopez',
                'gonzalez', 'perez', 'sanchez', 'ramirez', 'torres',
                'flores', 'rivera', 'gomez', 'diaz', 'reyes',
                'cruz', 'morales', 'ortiz', 'gutierrez', 'chavez'
            ]
        }
        
        # Check for each name with word boundaries
        for race, names in self.race_name_patterns.items():
            for name in names:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text_lower):
                    return race
        
        return 'unknown'
    
    def infer_race_from_text(self, text):
        """Infer race from cultural/educational patterns in anonymized text"""
        text_lower = text.lower()
        
        # Cultural/educational patterns
        cultural_patterns = {
            'black': ['hbu', 'hbcu', 'historically black', 'african american',
                     'urban league', 'naacp', 'black student union'],
            'asian': ['stem', 'technology institute', 'engineering school',
                     'massachusetts institute', 'caltech', 'carnegie mellon',
                     'asian american', 'confucius'],
            'hispanic': ['hispanic', 'latino', 'chicano', 'spanish', 'esl',
                        'migrant', 'border', 'texas rio grande'],
            'white': ['ivy league', 'preparatory school', 'boarding school',
                     'legacy', 'endowment', 'country club']
        }
        
        scores = {'black': 0, 'white': 0, 'asian': 0, 'hispanic': 0}
        
        for race, patterns in cultural_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    scores[race] += 1
        
        # Default to majority if no signals
        if max(scores.values()) == 0:
            return 'white'  # Default majority in US context
        
        return max(scores, key=scores.get)
    
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
    
    def infer_demographics(self, text):
        """Comprehensive demographic inference from text"""
        text_lower = text.lower()
        
        # Gender inference with multiple signals
        gender = self.infer_gender(text)
        
        # Race inference with multiple strategies
        race = self.infer_race_from_names(text)
        if race == 'unknown':
            race = self.infer_race_from_text(text)
        
        # Age group inference
        age_group = self.infer_age_group(text)
        
        # Add confidence scores
        return {
            'gender': gender,
            'race': race,
            'age_group': age_group,
            'gender_confidence': self._gender_confidence(text_lower, gender),
            'race_confidence': self._race_confidence(text_lower, race),
            'age_confidence': self._age_confidence(text_lower, age_group)
        }

    def _gender_confidence(self, text_lower, inferred_gender):
        """Calculate confidence for gender inference"""
        score = 0
        total_signals = 0
        
        # Pronoun signals
        for pronoun in [' he ', ' him ', ' his ']:
            if pronoun in text_lower:
                score += 1
                total_signals += 1
        
        for pronoun in [' she ', ' her ', ' hers ']:
            if pronoun in text_lower:
                score -= 1
                total_signals += 1
        
        # Name signals (simple check)
        if 'darnell' in text_lower or 'james' in text_lower or 'michael' in text_lower:
            score += 2
            total_signals += 2
        
        if 'emily' in text_lower or 'maria' in text_lower or 'jennifer' in text_lower:
            score -= 2
            total_signals += 2
        
        # Job keyword signals
        for keyword in self.job_gender_patterns['male_leaning']:
            if keyword in text_lower:
                score += 0.5
                total_signals += 0.5
        
        for keyword in self.job_gender_patterns['female_leaning']:
            if keyword in text_lower:
                score -= 0.5
                total_signals += 0.5
        
        if total_signals == 0:
            return 0.5  # Neutral confidence
        
        # Normalize confidence
        confidence = min(abs(score) / max(total_signals, 1), 1.0)
        
        # Adjust based on inference match
        if (inferred_gender == 'male' and score > 0) or (inferred_gender == 'female' and score < 0):
            return max(confidence, 0.7)
        else:
            return min(confidence, 0.6)

    def _race_confidence(self, text_lower, inferred_race):
        """Calculate confidence for race inference"""
        score = 0
        total_signals = 0
        
        # Name-based signals
        for race, names in self.race_name_patterns.items():
            for name in names:
                if name in text_lower:
                    if race == inferred_race:
                        score += 2
                    else:
                        score -= 1
                    total_signals += 2
        
        # Cultural pattern signals
        cultural_patterns = {
            'black': ['hbu', 'hbcu', 'historically black', 'african american'],
            'asian': ['mit', 'stanford', 'engineering', 'computer science'],
            'hispanic': ['spanish', 'bilingual', 'latino', 'hispanic'],
            'white': ['ivy league', 'preparatory', 'country club', 'legacy']
        }
        
        for race, patterns in cultural_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    if race == inferred_race:
                        score += 1
                    total_signals += 1
        
        if total_signals == 0:
            return 0.5
        
        confidence = min(abs(score) / max(total_signals, 1), 1.0)
        return confidence

    def _age_confidence(self, text_lower, inferred_age):
        """Calculate confidence for age inference"""
        score = 0
        total_signals = 0
        
        # Experience years
        exp_match = re.search(r'(\d+)\s+years', text_lower)
        if exp_match:
            years = int(exp_match.group(1))
            total_signals += 1
            if (inferred_age == 'early_career' and years <= 3) or \
              (inferred_age == 'mid_career' and 3 < years <= 10) or \
              (inferred_age == 'senior' and years > 10):
                score += 1
        
        # Graduation years
        grad_match = re.search(r'(?:class of|graduated)\s+(\d{4})', text_lower)
        if grad_match:
            total_signals += 1
            # Would need current year to calculate age
            score += 0.5  # Partial credit for having a year
        
        # Age phrases
        age_phrases = {
            'early_career': ['recent graduate', 'entry level', 'new grad', 'junior'],
            'mid_career': ['experienced', 'mid-level', 'professional'],
            'senior': ['senior', 'veteran', 'expert', 'lead', 'director']
        }
        
        for age_group, phrases in age_phrases.items():
            for phrase in phrases:
                if phrase in text_lower:
                    total_signals += 0.5
                    if age_group == inferred_age:
                        score += 0.5
        
        if total_signals == 0:
            return 0.5
        
        confidence = min(score / max(total_signals, 1), 1.0)
        return confidence


class FairnessMetrics:
    """Compute fairness metrics for multi-class classification"""
    
    @staticmethod
    def demographic_parity_difference(y_pred, protected_attr):
        """Calculate demographic parity difference"""
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


class NameSubstitutionExperiment:
    """Name substitution experiments for bias measurement"""
    
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
        self.male_names = ['James', 'Robert', 'John', 'Michael', 'David', 'William', 
                          'Richard', 'Joseph', 'Thomas', 'Christopher', 'Daniel', 'Matthew']
        self.female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 
                            'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa']
        
        self.white_names = ['Emily', 'Anne', 'Jill', 'Allison', 'Laurie', 'Neil', 
                           'Geoffrey', 'Brett', 'Greg', 'Matthew', 'Katie', 'Megan']
        self.black_names = ['Lakisha', 'Latoya', 'Tamika', 'Imani', 'Ebony', 'Darnell',
                           'Jermaine', 'Tyrone', 'DeShawn', 'Marquis', 'Shanice', 'Aaliyah']
        self.asian_names = ['Wei', 'Jing', 'Li', 'Zhang', 'Wang', 'Chen', 
                           'Yong', 'Min', 'Hui', 'Xiao', 'Mei', 'Lin']
        self.hispanic_names = ['Jose', 'Carlos', 'Luis', 'Juan', 'Miguel', 'Rosa',
                              'Maria', 'Carmen', 'Ana', 'Dolores', 'Sofia', 'Isabella']
    
    def run_bias_experiment(self, test_texts, test_labels, num_samples=50):
        """Run bias experiments with name substitution"""
        
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
                male_text = f"Candidate: {np.random.choice(self.male_names)} {text}"
                male_pred, _ = self._predict_single(male_text)
                
                female_text = f"Candidate: {np.random.choice(self.female_names)} {text}"
                female_pred, _ = self._predict_single(female_text)
                
                male_predictions.append(male_pred)
                female_predictions.append(female_pred)
                
                bias_score = 1 if male_pred != female_pred else 0
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
        racial_groups = {
            'white': self.white_names,
            'black': self.black_names, 
            'asian': self.asian_names,
            'hispanic': self.hispanic_names
        }
        
        group_predictions = {group: [] for group in racial_groups.keys()}
        
        for i, (text, label) in enumerate(zip(test_texts[:num_samples], test_labels[:num_samples])):
            try:
                for group, names in racial_groups.items():
                    group_text = f"Candidate: {np.random.choice(names)} {text}"
                    pred, _ = self._predict_single(group_text)
                    group_predictions[group].append(pred)
                
            except Exception as e:
                print(f"Error in racial bias experiment: {e}")
                continue
        
        return {
            'group_predictions': {k: (np.mean(v) if v else 0) for k, v in group_predictions.items()},
            'average_bias': self._calculate_average_bias(group_predictions),
            'white_black_disparity': np.mean(group_predictions['white']) - np.mean(group_predictions['black']) if group_predictions['white'] and group_predictions['black'] else 0
        }
    
    def _calculate_average_bias(self, group_predictions):
        """Calculate average bias across all groups"""
        group_means = []
        for group, preds in group_predictions.items():
            if preds:
                group_means.append(np.mean(preds))
        
        if len(group_means) >= 2:
            return max(group_means) - min(group_means)
        return 0
    
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


class BiasAnalyzer:
    """Comprehensive bias analysis framework"""
    
    def __init__(self, model, tokenizer, label_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.device = device
        self.demographic_inference = DemographicInference()
        self.name_experiment = NameSubstitutionExperiment(tokenizer, model, device)
        self.fairness_metrics = FairnessMetrics()
        self.lime_explainer = LimeExplainer(model, tokenizer, label_map, device)
        
    def bias_analysis(self, test_texts, test_labels, test_predictions):
        """Run comprehensive bias analysis"""
        
        test_labels = np.array(test_labels)
        test_predictions = np.array(test_predictions)
        
        demographics = self._infer_demographics(test_texts)
        
        fairness_results = self._calculate_fairness_metrics(test_labels, test_predictions, demographics)
        
        category_bias = self._analyze_category_bias(test_labels, test_predictions, demographics)
        
        try:
            name_bias_results = self.name_experiment.run_bias_experiment(test_texts, test_labels, num_samples=30)
        except Exception as e:
            print(f"Name substitution experiment failed: {e}")
            name_bias_results = {
                'gender_bias': {'average_bias': 0.0, 'male_female_disparity': 0.0},
                'racial_bias': {'average_bias': 0.0, 'white_black_disparity': 0.0}
            }
        
        # SKIP LIME for now - it's not critical for bias measurement
        lime_explanations = []  # Empty list instead of calling _generate_lime_explanations
        
        report = self._generate_bias_report(
            fairness_results, category_bias, name_bias_results, lime_explanations,
            test_labels, test_predictions
        )
        
        return report
    
    def _infer_demographics(self, texts):
        """Infer demographics from texts"""
        demographics = {
            'gender': [],
            'race': [],
            'age_group': []
        }
        
        for text in texts:
            # Use the new infer_demographics method
            demo = self.demographic_inference.infer_demographics(text)
            demographics['gender'].append(demo['gender'])
            demographics['race'].append(demo['race'])
            demographics['age_group'].append(demo['age_group'])
        
        return demographics
    
    def _calculate_fairness_metrics(self, y_true, y_pred, demographics):
        """Calculate fairness metrics with demographic attributes"""
        metrics = {}
        
        for demo_type, demo_values in demographics.items():
            demo_encoder = LabelEncoder()
            demo_encoded = demo_encoder.fit_transform(demo_values)
            
            try:
                metrics[demo_type] = {
                    'demographic_parity': self._demographic_parity_difference(y_pred, demo_encoded),
                    'equal_opportunity': self._equal_opportunity_difference(y_true, y_pred, demo_encoded),
                    'accuracy_equality': self._accuracy_equality_difference(y_true, y_pred, demo_encoded)
                }
            except Exception as e:
                print(f"Error calculating {demo_type} metrics: {e}")
                metrics[demo_type] = {
                    'demographic_parity': 0.0,
                    'equal_opportunity': 0.0,
                    'accuracy_equality': 0.0
                }
        
        return metrics
    
    def _demographic_parity_difference(self, y_pred, protected_attr):
        """Calculate demographic parity for multi-class classification"""
        protected_attr = np.array(protected_attr)
        y_pred = np.array(y_pred)
        
        groups = np.unique(protected_attr)
        unique_classes = np.unique(y_pred)
        max_differences = []
        
        for class_label in unique_classes:
            group_positive_rates = []
            for group in groups:
                group_mask = protected_attr == group
                if np.sum(group_mask) > 0:
                    positive_rate = np.mean(y_pred[group_mask] == class_label)
                    group_positive_rates.append(positive_rate)
            
            if group_positive_rates:
                max_diff = max(group_positive_rates) - min(group_positive_rates)
                max_differences.append(max_diff)
        
        return max(max_differences) if max_differences else 0
    
    def _equal_opportunity_difference(self, y_true, y_pred, protected_attr):
        """Calculate equal opportunity difference for multi-class classification"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        groups = np.unique(protected_attr)
        num_classes = len(np.unique(y_true))
        
        # Calculate TPR for each group and each class
        max_differences = []
        
        for class_label in range(num_classes):
            group_tprs = []
            
            for group in groups:
                group_mask = protected_attr == group
                
                # True positives: samples of this class correctly predicted
                tp_mask = group_mask & (y_true == class_label) & (y_pred == class_label)
                tp = np.sum(tp_mask)
                
                # Actual positives: samples of this class in the group
                actual_positives = np.sum(group_mask & (y_true == class_label))
                
                if actual_positives > 0:
                    tpr = tp / actual_positives
                else:
                    tpr = 0  # No samples of this class in this group
                
                group_tprs.append(tpr)
            
            # Calculate difference for this class
            if len(group_tprs) >= 2:
                diff = max(group_tprs) - min(group_tprs)
                max_differences.append(diff)
        
        # Return average difference across classes
        return np.mean(max_differences) if max_differences else 0
    
    def _accuracy_equality_difference(self, y_true, y_pred, protected_attr):
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
    
    def _analyze_category_bias(self, y_true, y_pred, demographics):
        """Analyze bias across job categories"""
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
    
    def _generate_lime_explanations(self, texts, labels):
        """Generate LIME explanations for sample texts"""
        explanations = []
        
        for i, (text, true_label) in enumerate(zip(texts, labels)):
            try:
                exp = self.lime_explainer.explain_prediction(text, num_features=8)
                if exp is not None:
                    explanation_data = {
                        'text_preview': text[:100] + '...' if len(text) > 100 else text,
                        'true_label': self.label_map.get(str(true_label), f"Category_{true_label}"),
                        'top_features': exp.as_list()[:5] if hasattr(exp, 'as_list') else []
                    }
                    explanations.append(explanation_data)
            except Exception as e:
                print(f"LIME explanation failed for sample {i}: {e}")
                continue
        
        return explanations
    
    def _calculate_category_bias_score(self, demo_analysis):
        """Calculate overall bias score for a category"""
        max_disparities = []
        
        for demo_type, analysis in demo_analysis.items():
            if 'max_disparity' in analysis:
                max_disparities.append(analysis['max_disparity'])
        
        return np.mean(max_disparities) if max_disparities else 0.0
    
    def _generate_bias_report(self, fairness_results, category_bias, name_bias_results, lime_explanations, y_true, y_pred):
        """Generate comprehensive bias analysis report"""
        
        report = {
            'fairness_metrics': fairness_results,
            'category_bias_analysis': category_bias,
            'name_substitution_bias': name_bias_results,
            'lime_explanations': lime_explanations,
            'overall_performance': {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
            },
            'recommendations': self._generate_recommendations(
                fairness_results, category_bias, name_bias_results
            )
        }
        
        self._print_bias_summary(report)
        
        return report
    
    def _generate_recommendations(self, fairness_results, category_bias, name_bias_results):
        """Generate actionable recommendations"""
        recommendations = []
        
        high_bias_threshold = 0.1
        medium_bias_threshold = 0.05
        
        for demo_type, metrics in fairness_results.items():
            dp_score = metrics.get('demographic_parity', 0)
            eo_score = metrics.get('equal_opportunity', 0)
            
            if dp_score > high_bias_threshold:
                recommendations.append(
                    f"High demographic parity disparity ({dp_score:.3f}) for {demo_type}. "
                    f"Implement preprocessing and reweighting strategies."
                )
            elif dp_score > medium_bias_threshold:
                recommendations.append(
                    f"Moderate demographic parity disparity ({dp_score:.3f}) for {demo_type}. "
                    f"Consider dataset balancing."
                )
            
            if eo_score > high_bias_threshold:
                recommendations.append(
                    f"Equal opportunity concerns ({eo_score:.3f}) for {demo_type}. "
                    f"Implement adversarial debiasing."
                )
        
        gender_bias = name_bias_results.get('gender_bias', {}).get('average_bias', 0)
        racial_bias = name_bias_results.get('racial_bias', {}).get('average_bias', 0)
        
        if gender_bias > 0.05:
            recommendations.append(
                f"Gender bias detected ({gender_bias:.3f}). "
                f"Remove names during preprocessing."
            )
        
        if racial_bias > 0.05:
            recommendations.append(
                f"Racial bias detected ({racial_bias:.3f}). "
                f"Implement comprehensive debiasing pipeline."
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
                f"High bias and low accuracy in {worst_category[0]} "
                f"(bias: {worst_category[1]:.3f}, accuracy: {worst_category[2]:.3f}). "
                f"Focus debiasing on this category."
            )
        
        if not recommendations:
            recommendations = [
                "System shows generally fair performance. Continue monitoring.",
                "Implement regular bias audits.",
                "Consider proactive debiasing for continuous improvement."
            ]
        
        return recommendations[:5]
    
    def _print_bias_summary(self, report):
        """Print bias analysis summary"""
        
        perf = report['overall_performance']
        print(f"Overall Performance:")
        print(f"  Accuracy: {perf['accuracy']:.3f}")
        print(f"  F1 Score: {perf['f1']:.3f}")
        
        print(f"\nFairness Metrics:")
        for demo_type, metrics in report['fairness_metrics'].items():
            print(f"  {demo_type}: ")
            print(f"    Demographic Parity: {metrics['demographic_parity']:.3f}")
            print(f"    Equal Opportunity: {metrics['equal_opportunity']:.3f}")
            print(f"    Accuracy Equality: {metrics['accuracy_equality']:.3f}")
        
        name_bias = report['name_substitution_bias']
        print(f"\nName-based Bias:")
        print(f"  Gender Bias: {name_bias['gender_bias']['average_bias']:.3f}")
        print(f"  Racial Bias: {name_bias['racial_bias']['average_bias']:.3f}")
        
        if report.get('lime_explanations'):
            print(f"\nLIME Explanations Generated: {len(report['lime_explanations'])} samples")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
