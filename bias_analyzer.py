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
    """Demographic inference from resume text - FIXED VERSION"""
    
    def __init__(self):
        # More comprehensive gender patterns
        self.gender_patterns = {
            'male': [
                r'\bhe\b', r'\bhim\b', r'\bhis\b', r'\bmale\b', r'\bman\b', r'\bmen\b',
                r'\bboy\b', r'\bmr\.', r'\bmr\b', r'\bmister\b', r'\bfather\b', r'\bhusband\b',
                r'\bbrother\b', r'\bson\b', r'\bgentleman\b', r'\bmale\b',
                r'\bguy\b', r'\bsir\b', r'\bmale-identifying\b', r'\bmasculine\b'
            ],
            'female': [
                r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bfemale\b', r'\bwoman\b', r'\bwomen\b',
                r'\bgirl\b', r'\bms\.', r'\bms\b', r'\bmiss\b', r'\bmrs\.', r'\bmrs\b',
                r'\bmother\b', r'\bwife\b', r'\bsister\b', r'\bdaughter\b', r'\blady\b',
                r'\bfemale\b', r'\bgal\b', r'\bmadam\b', r'\bfemale-identifying\b', r'\bfeminine\b'
            ]
        }
        
        # Enhanced race/ethnicity name patterns
        self.race_name_patterns = {
            'black': [
                'darnell', 'lakisha', 'latoya', 'tamika', 'imani', 'ebony', 
                'jermaine', 'tyrone', 'deshawn', 'marquis', 'shanice', 'aaliyah',
                'keisha', 'jamal', 'latonya', 'tyrell', 'shaniqua', 'deandre',
                'tanisha', 'malik', 'kareem', 'darius', 'lashonda', 'precious',
                'tremayne', 'quintavius', 'lakendra', 'tamiko', 'desiree'
            ],
            'white': [
                'emily', 'anne', 'jill', 'allison', 'laurie', 'neil',
                'geoffrey', 'brett', 'greg', 'matthew', 'katie', 'megan',
                'james', 'robert', 'john', 'michael', 'david', 'william',
                'richard', 'joseph', 'thomas', 'christopher', 'daniel',
                'mary', 'patricia', 'jennifer', 'linda', 'elizabeth', 'sarah',
                'jessica', 'susan', 'nancy', 'lisa', 'karen', 'betty'
            ],
            'asian': [
                'chen', 'wei', 'jing', 'li', 'zhang', 'wang', 'liu',
                'yong', 'min', 'hui', 'xiao', 'mei', 'lin', 'yang',
                'kim', 'park', 'choi', 'lee', 'jung', 'kang', 'cho',
                'tanaka', 'sato', 'suzuki', 'takahashi', 'watanabe', 'yamamoto',
                'patel', 'sharma', 'kumar', 'singh', 'gupta', 'shah'
            ],
            'hispanic': [
                'garcia', 'rodriguez', 'martinez', 'hernandez', 'lopez',
                'gonzalez', 'perez', 'sanchez', 'ramirez', 'torres',
                'flores', 'rivera', 'gomez', 'diaz', 'reyes', 'cruz',
                'morales', 'ortiz', 'gutierrez', 'chavez', 'ruiz', 'alvarez',
                'castillo', 'romero', 'vargas', 'medina', 'aguilar', 'herrera'
            ]
        }
        
        # Cultural/context patterns for anonymized resumes
        self.cultural_patterns = {
            'black': [
                r'\bhbcu\b', r'\bhistorically black\b', r'\bafrican american\b',
                r'\bnaacp\b', r'\bblack student union\b', r'\bblack engineers\b',
                r'\burban league\b', r'\bblack in\b', r'\bafrican diaspora\b',
                r'\bcivil rights\b', r'\bsocial justice\b', r'\bcommunity organizer\b'
            ],
            'white': [
                r'\bivy league\b', r'\bpreparatory school\b', r'\bboarding school\b',
                r'\bcountry club\b', r'\blegacy\b', r'\bendowment\b',
                r'\byacht club\b', r'\bpolo\b', r'\bgolf club\b', r'\btennis club\b',
                r'\bprivate school\b', r'\bprep school\b', r'\bupper east side\b'
            ],
            'asian': [
                r'\bstem\b', r'\bmit\b', r'\bcaltech\b', r'\bcarnegie mellon\b',
                r'\bengineering school\b', r'\bmath olympiad\b', r'\bscience olympiad\b',
                r'\bprogramming competition\b', r'\bhackathon\b', r'\brobotics\b',
                r'\bconfucius\b', r'\basian american\b', r'\bmodel minority\b'
            ],
            'hispanic': [
                r'\bhispanic\b', r'\blatino\b', r'\blatina\b', r'\bchicano\b',
                r'\bspanish speaking\b', r'\besl\b', r'\bbilingual\b',
                r'\bmigrant\b', r'\bimmigrant\b', r'\bborder\b', r'\btexas rio grande\b',
                r'\bunidos\b', r'\bfiesta\b', r'\bquinceañera\b', r'\bdia de los muertos\b'
            ]
        }
        
        # Age group patterns
        self.age_patterns = {
            'early_career': [
                r'\brecent graduate\b', r'\bentry level\b', r'\bnew grad\b',
                r'\bjunior\b', r'\bassociate\b', r'\bintern\b',
                r'\bclass of (202[0-4]|201[7-9])\b',  # Recent graduates
                r'\b(0|1|2|3)\s+years?\s+experience\b',
                r'\bfresher\b', r'\btrainee\b'
            ],
            'mid_career': [
                r'\bmid-level\b', r'\bprofessional\b', r'\bexperienced\b',
                r'\b(4|5|6|7|8|9|10)\s+years?\s+experience\b',
                r'\bsenior\s+associate\b', r'\bmanager\b',
                r'\bclass of (201[0-6]|200[0-9])\b'  # Graduated 2010-2016
            ],
            'senior': [
                r'\bsenior\b', r'\bprincipal\b', r'\bdirector\b',
                r'\bvice president\b', r'\bvp\b', r'\bexecutive\b',
                r'\b(1[1-9]|[2-9]\d+)\s+years?\s+experience\b',  # 11+ years
                r'\bclass of (19\d{2}|200[0-9])\b',  # Graduated before 2010
                r'\bveteran\b', r'\bexpert\b', r'\blead\b', r'\bhead of\b'
            ]
        }
        
    def infer_gender(self, text):
        """Infer gender from text with improved pattern matching"""
        text_lower = text.lower()
        
        male_score = 0
        female_score = 0
        
        # Check direct patterns first
        for pattern in self.gender_patterns['male']:
            matches = re.findall(pattern, text_lower)
            male_score += len(matches)
        
        for pattern in self.gender_patterns['female']:
            matches = re.findall(pattern, text_lower)
            female_score += len(matches)
        
        # Additional heuristic: "his" vs "her" frequency
        his_count = len(re.findall(r'\bhis\b', text_lower))
        her_count = len(re.findall(r'\bher\b', text_lower))
        male_score += his_count * 2
        female_score += her_count * 2
        
        # Determine gender
        if male_score > female_score:
            return 'male'
        elif female_score > male_score:
            return 'female'
        else:
            # Check for common gender-neutral patterns
            if re.search(r'\bthey\b|\bthem\b|\btheir\b', text_lower):
                return 'non-binary'
            return 'unknown'
    
    def infer_race_from_names(self, text):
        """Infer race from names with improved matching"""
        text_lower = text.lower()
        
        race_scores = {'black': 0, 'white': 0, 'asian': 0, 'hispanic': 0}
        
        # Check names
        for race, names in self.race_name_patterns.items():
            for name in names:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text_lower):
                    race_scores[race] += 2
        
        # Check cultural patterns
        for race, patterns in self.cultural_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    race_scores[race] += 1
        
        # Return race with highest score
        if sum(race_scores.values()) > 0:
            return max(race_scores, key=race_scores.get)
        
        return 'unknown'
    
    def infer_race_from_text(self, text):
        """Infer race from text patterns (for anonymized resumes)"""
        text_lower = text.lower()
        
        race_scores = {'black': 0, 'white': 0, 'asian': 0, 'hispanic': 0}
        
        # Check cultural patterns
        for race, patterns in self.cultural_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    race_scores[race] += 2
        
        # Education patterns
        if re.search(r'\bhbcu|howard|spelman|morehouse|hampton\b', text_lower):
            race_scores['black'] += 3
        
        if re.search(r'\bivy league|harvard|yale|princeton|stanford\b', text_lower):
            race_scores['white'] += 2
        
        if re.search(r'\bmit|caltech|carnegie mellon|engineering school\b', text_lower):
            race_scores['asian'] += 2
        
        if re.search(r'\bspanish|bilingual|esl|migrant|immigrant\b', text_lower):
            race_scores['hispanic'] += 2
        
        # Return highest scoring race
        if sum(race_scores.values()) > 0:
            return max(race_scores, key=race_scores.get)
        
        return 'unknown'
    
    def infer_age_group(self, text):
        """Infer age group from text patterns"""
        text_lower = text.lower()
        
        age_scores = {'early_career': 0, 'mid_career': 0, 'senior': 0}
        
        # Check age patterns
        for age_group, patterns in self.age_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                age_scores[age_group] += len(matches)
        
        # Check for years of experience
        years_exp_match = re.search(r'(\d+)\s+years?\s+experience', text_lower)
        if years_exp_match:
            years = int(years_exp_match.group(1))
            if years <= 3:
                age_scores['early_career'] += 2
            elif years <= 10:
                age_scores['mid_career'] += 2
            else:
                age_scores['senior'] += 2
        
        # Check graduation year
        grad_match = re.search(r'class of\s+(\d{4})', text_lower, re.IGNORECASE)
        if grad_match:
            grad_year = int(grad_match.group(1))
            current_year = 2024  # Update this as needed
            years_since_grad = current_year - grad_year
            if years_since_grad <= 3:
                age_scores['early_career'] += 2
            elif years_since_grad <= 15:
                age_scores['mid_career'] += 2
            else:
                age_scores['senior'] += 2
        
        # Return age group with highest score
        if sum(age_scores.values()) > 0:
            return max(age_scores, key=age_scores.get)
        
        return 'unknown'
    
    def infer_demographics(self, text):
        """Comprehensive demographic inference"""
        gender = self.infer_gender(text)
        
        # Try name-based inference first
        race = self.infer_race_from_names(text)
        if race == 'unknown':
            race = self.infer_race_from_text(text)
        
        # Age inference
        age_group = self.infer_age_group(text)
        
        return {
            'gender': gender,
            'race': race,
            'age_group': age_group
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
    """Name substitution experiments for bias measurement - FIXED"""
    
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        
        # More diverse name sets
        self.male_names = ['James', 'Robert', 'John', 'Michael', 'David', 'William', 
                          'Richard', 'Joseph', 'Thomas', 'Christopher', 'Daniel', 'Matthew',
                          'Anthony', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua',
                          'Kenneth', 'Kevin', 'Brian', 'George', 'Edward', 'Ronald']
        
        self.female_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 
                            'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa',
                            'Margaret', 'Sandra', 'Ashley', 'Kimberly', 'Emily', 'Donna',
                            'Michelle', 'Dorothy', 'Carol', 'Amanda', 'Melissa', 'Deborah']
        
        # Race-specific names with clear cultural associations
        self.white_names = ['Emily', 'Anne', 'Jill', 'Allison', 'Laurie', 'Neil', 
                           'Geoffrey', 'Brett', 'Greg', 'Matthew', 'Katie', 'Megan',
                           'Heather', 'Tiffany', 'Amber', 'Stephanie', 'Brittany', 'Courtney']
        
        self.black_names = ['Lakisha', 'Latoya', 'Tamika', 'Imani', 'Ebony', 'Darnell',
                           'Jermaine', 'Tyrone', 'DeShawn', 'Marquis', 'Shanice', 'Aaliyah',
                           'Keisha', 'Jamal', 'Latonya', 'Tyrone', 'DeAndre', 'Shaniqua']
        
        self.asian_names = ['Wei', 'Jing', 'Li', 'Zhang', 'Wang', 'Chen', 
                           'Yong', 'Min', 'Hui', 'Xiao', 'Mei', 'Lin',
                           'Kim', 'Park', 'Choi', 'Lee', 'Jung', 'Kang']
        
        self.hispanic_names = ['Jose', 'Carlos', 'Luis', 'Juan', 'Miguel', 'Rosa',
                              'Maria', 'Carmen', 'Ana', 'Dolores', 'Sofia', 'Isabella',
                              'Jesus', 'Francisco', 'Jorge', 'Pedro', 'Manuel', 'Ricardo']
    
    def _predict_single(self, text):
        """Predict class and confidence for a single text"""
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
                confidence, pred = torch.max(probs, dim=-1)
                return pred.cpu().item(), confidence.cpu().item()
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0, 0.0  # Default to class 0 with 0 confidence
    
    def _run_gender_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure gender bias through name substitution - FIXED"""
        male_predictions = []
        female_predictions = []
        bias_scores = []
        
        samples_to_test = min(num_samples, len(test_texts))
        
        for i in range(samples_to_test):
            text = test_texts[i]
            label = test_labels[i]
            
            try:
                # Create versions with different gendered names
                # Insert at the beginning where it's most noticeable
                male_text = f"Applicant: {np.random.choice(self.male_names)}. {text}"
                male_pred, male_conf = self._predict_single(male_text)
                
                female_text = f"Applicant: {np.random.choice(self.female_names)}. {text}"
                female_pred, female_conf = self._predict_single(female_text)
                
                # Also test with pronouns added
                male_text_with_pronouns = f"Applicant: {np.random.choice(self.male_names)}. He has experience in {text[:100]}..."
                male_pred2, male_conf2 = self._predict_single(male_text_with_pronouns)
                
                female_text_with_pronouns = f"Applicant: {np.random.choice(self.female_names)}. She has experience in {text[:100]}..."
                female_pred2, female_conf2 = self._predict_single(female_text_with_pronouns)
                
                male_predictions.extend([male_pred, male_pred2])
                female_predictions.extend([female_pred, female_pred2])
                
                # Calculate bias score (1 if predictions differ)
                bias_score = 1 if male_pred != female_pred or male_pred2 != female_pred2 else 0
                bias_scores.append(bias_score)
                
            except Exception as e:
                print(f"Error in gender bias experiment sample {i}: {e}")
                continue
        
        if not male_predictions or not female_predictions:
            return {
                'male_predictions': [],
                'female_predictions': [],
                'bias_scores': [],
                'average_bias': 0.0,
                'male_female_disparity': 0.0
            }
        
        return {
            'male_predictions': male_predictions,
            'female_predictions': female_predictions,
            'bias_scores': bias_scores,
            'average_bias': np.mean(bias_scores) if bias_scores else 0.0,
            'male_female_disparity': np.mean(male_predictions) - np.mean(female_predictions) if male_predictions and female_predictions else 0.0
        }
    
    def _run_racial_bias_experiment(self, test_texts, test_labels, num_samples):
        """Measure racial bias through name substitution - FIXED"""
        racial_groups = {
            'white': self.white_names,
            'black': self.black_names, 
            'asian': self.asian_names,
            'hispanic': self.hispanic_names
        }
        
        group_predictions = {group: [] for group in racial_groups.keys()}
        group_confidences = {group: [] for group in racial_groups.keys()}
        
        samples_to_test = min(num_samples, len(test_texts))
        
        for i in range(samples_to_test):
            text = test_texts[i]
            label = test_labels[i]
            
            try:
                for group, names in racial_groups.items():
                    # Create text with race-associated name
                    group_text = f"Applicant: {np.random.choice(names)}. {text}"
                    pred, conf = self._predict_single(group_text)
                    
                    group_predictions[group].append(pred)
                    group_confidences[group].append(conf)
                
            except Exception as e:
                print(f"Error in racial bias experiment sample {i}: {e}")
                continue
        
        # Calculate statistics
        results = {
            'group_predictions': {},
            'group_confidences': {},
            'average_bias': 0.0,
            'white_black_disparity': 0.0,
            'prediction_distribution': {}
        }
        
        for group, preds in group_predictions.items():
            if preds:
                results['group_predictions'][group] = {
                    'mean_prediction': float(np.mean(preds)),
                    'std_prediction': float(np.std(preds)),
                    'count': len(preds)
                }
                
                if group in group_confidences and group_confidences[group]:
                    results['group_confidences'][group] = {
                        'mean_confidence': float(np.mean(group_confidences[group])),
                        'std_confidence': float(np.std(group_confidences[group]))
                    }
        
        # Calculate bias metrics
        group_means = []
        for group, data in results['group_predictions'].items():
            group_means.append(data['mean_prediction'])
        
        if len(group_means) >= 2:
            results['average_bias'] = float(max(group_means) - min(group_means))
        
        # Calculate white-black disparity specifically
        if 'white' in results['group_predictions'] and 'black' in results['group_predictions']:
            white_mean = results['group_predictions']['white']['mean_prediction']
            black_mean = results['group_predictions']['black']['mean_prediction']
            results['white_black_disparity'] = float(white_mean - black_mean)
        
        return results
    
    def run_bias_experiment(self, test_texts, test_labels, num_samples=30):
        """Run both gender and racial bias experiments"""
        gender_bias = self._run_gender_bias_experiment(test_texts, test_labels, num_samples)
        racial_bias = self._run_racial_bias_experiment(test_texts, test_labels, num_samples)
        
        return {
            'gender_bias': gender_bias,
            'racial_bias': racial_bias
        }

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
