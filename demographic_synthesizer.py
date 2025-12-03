"""
Demographic signal synthesizer for debiasing training.
"""

import re
import random
from typing import List, Tuple
import numpy as np


class DemographicSynthesizer:
    """Synthesize demographic signals for adversarial training"""
    
    def __init__(self):
        # Gender-related keywords that might appear in anonymized resumes
        self.gender_keywords = {
            'male_biased': [
                'football', 'basketball', 'baseball', 'golf', 'fishing', 'hunting',
                'military', 'engineering', 'construction', 'programming', 'coding',
                'software engineer', 'mechanical engineer', 'electrical engineer'
            ],
            'female_biased': [
                'nursing', 'teaching', 'childcare', 'counseling', 'social work',
                'human resources', 'public relations', 'event planning', 'interior design',
                'early childhood education', 'family therapy', 'community outreach'
            ]
        }
        
        # Racial/ethnic bias signals
        self.cultural_keywords = {
            'black_culture': [
                'HBCU', 'historically black college', 'NAACP', 'urban league',
                'community organizer', 'social justice', 'diversity initiative',
                'inclusion committee', 'multicultural affairs'
            ],
            'asian_culture': [
                'STEM', 'mathematics', 'physics', 'computer science', 'engineering',
                'technology conference', 'coding bootcamp', 'hackathon', 'robotics'
            ],
            'hispanic_culture': [
                'bilingual', 'spanish', 'latino', 'hispanic', 'community center',
                'immigrant services', 'ESL', 'english as second language'
            ]
        }
        
        # Socioeconomic signals
        self.class_keywords = {
            'upper_class': [
                'ivy league', 'stanford', 'mit', 'harvard', 'yale', 'princeton',
                'private school', 'preparatory school', 'country club',
                'summer internship abroad', 'study abroad program'
            ],
            'working_class': [
                'community college', 'state university', 'public school',
                'part-time job', 'worked through college', 'first generation',
                'evening classes', 'night shift', 'factory', 'retail'
            ]
        }
    
    def extract_demographic_signals(self, text: str) -> List[float]:
        """Extract demographic signal features from text"""
        text_lower = text.lower()
        
        features = []
        
        # Gender signals (2 features)
        male_score = sum(1 for word in self.gender_keywords['male_biased'] 
                        if word in text_lower)
        female_score = sum(1 for word in self.gender_keywords['female_biased'] 
                          if word in text_lower)
        
        # Normalize
        total_gender = male_score + female_score
        if total_gender > 0:
            features.append(male_score / total_gender)
            features.append(female_score / total_gender)
        else:
            features.extend([0.5, 0.5])  # Neutral
        
        # Cultural signals (3 features)
        for culture in ['black_culture', 'asian_culture', 'hispanic_culture']:
            score = sum(1 for word in self.cultural_keywords[culture] 
                       if word in text_lower)
            # Use log scale to prevent domination
            features.append(np.log1p(score))
        
        # Class signals (2 features)
        upper_score = sum(1 for word in self.class_keywords['upper_class'] 
                         if word in text_lower)
        working_score = sum(1 for word in self.class_keywords['working_class'] 
                           if word in text_lower)
        
        total_class = upper_score + working_score
        if total_class > 0:
            features.append(upper_score / total_class)
            features.append(working_score / total_class)
        else:
            features.extend([0.5, 0.5])
        
        return np.array(features, dtype=np.float32)
    
    def generate_protected_labels(self, features: np.ndarray) -> Tuple[int, int]:
        """Generate protected attribute labels from features"""
        # Gender label (0=male-leaning, 1=female-leaning)
        gender_feature = features[0] - features[1]  # male - female
        gender_label = 0 if gender_feature > 0.1 else 1 if gender_feature < -0.1 else random.randint(0, 1)
        
        # Privilege label (0=disadvantaged, 1=advantaged)
        # Based on cultural and class signals
        privilege_score = (np.mean(features[2:5]) + features[5]) / 2  # cultural + class
        privilege_label = 0 if privilege_score < 0.3 else 1 if privilege_score > 0.7 else random.randint(0, 1)
        
        return gender_label, privilege_label
