"""
Enhanced Data Processing Module for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import re
import os
import pandas as pd
import numpy as np
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import json
from collections import Counter


class ResumePreprocessor:
    """Enhanced text preprocessing for resumes with demographic signal removal"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s\.,;:!?\-]')
        
        # Demographic indicators to remove
        self.gender_indicators = [
            'he', 'she', 'him', 'her', 'his', 'hers',
            'mr', 'mrs', 'ms', 'miss', 'sir', 'madam',
            'father', 'mother', 'husband', 'wife',
            'brother', 'sister', 'son', 'daughter'
        ]
        
        self.privilege_indicators = [
            'harvard', 'stanford', 'mit', 'princeton', 'yale',
            'oxford', 'cambridge', 'ivy league', 'goldman sachs',
            'mckinsey', 'google', 'microsoft', 'apple'
        ]

    def clean_text(self, text):
        """Clean text with demographic signal removal"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        # Remove URLs, emails, and phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove demographic indicators
        for indicator in self.gender_indicators:
            text = re.sub(r'\b' + indicator + r'\b', ' ', text)
        
        for indicator in self.privilege_indicators:
            text = re.sub(r'\b' + indicator + r'\b', ' ', text)
        
        # Remove special characters
        text = self.special_chars.sub(' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_enhanced_features(self, text):
        """Extract job-related features for improved classification"""
        text_lower = text.lower()
        
        technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#'],
            'web': ['html', 'css', 'react', 'angular', 'vue'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb']
        }
        
        features = []
        
        for category, skills in technical_skills.items():
            found_skills = [skill for skill in skills if skill in text_lower]
            if found_skills:
                features.extend(found_skills)
        
        return ' '.join(features)


def download_dataset():
    """Download dataset from Google Drive"""
    os.makedirs('data/raw', exist_ok=True)
    
    if not os.path.exists('data/raw/Resume.csv'):
        print("Downloading dataset...")
        try:
            gdown.download(
                'https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk',
                'data/raw/Resume.csv', 
                quiet=False
            )
            print("Dataset downloaded successfully")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    else:
        print("Dataset already exists")
        return True


def enhanced_balance_dataset(df, target_col='Category'):
    """Balance dataset with targeted augmentation - FIXED VERSION"""
    print("Applying dataset balancing with targeted augmentation...")
    
    underperforming_categories = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA']
    
    # Get category distribution
    category_counts = df[target_col].value_counts()
    
    balanced_indices = []
    
    for category in category_counts.index:
        category_indices = df[df[target_col] == category].index.tolist()
        needed_count = max(category_counts.values())
        
        if len(category_indices) < needed_count:
            duplicates_needed = needed_count - len(category_indices)
            
            # For underperforming categories, use augmentation
            if category in underperforming_categories and duplicates_needed > 0:
                augmented_indices = []
                for i in range(duplicates_needed):
                    original_idx = np.random.choice(category_indices)
                    augmented_indices.append(original_idx)
                
                balanced_indices.extend(category_indices)
                balanced_indices.extend(augmented_indices)
            else:
                duplicated = np.random.choice(category_indices, duplicates_needed, replace=True)
                balanced_indices.extend(category_indices)
                balanced_indices.extend(duplicated.tolist())
        else:
            balanced_indices.extend(category_indices[:needed_count])
    
    balanced_df = df.iloc[balanced_indices].copy()
    balanced_df.reset_index(drop=True, inplace=True)
    
    print(f"Dataset balancing complete: {len(df)} -> {len(balanced_df)} samples")
    
    return balanced_df


def analyze_and_fix_class_imbalance(df, category_col='Category'):
    """Comprehensive class imbalance analysis and fixing - FIXED VERSION"""
    category_counts = df[category_col].value_counts()
    
    print("Class Imbalance Analysis & Fixing")
    print("=" * 60)
    
    problem_classes = category_counts[category_counts < 10]
    if len(problem_classes) > 0:
        print(f"Critical: {len(problem_classes)} classes have < 10 samples:")
        for cls, count in problem_classes.items():
            print(f"  {cls}: {count} samples")
        
        df_balanced = enhanced_balance_dataset(df, category_col)
        return df_balanced
    else:
        print("No critical class imbalance detected")
        return df


def load_and_preprocess_data(data_path):
    """Load and preprocess resume data - FIXED VERSION"""
    print("Enhanced Data Processing")
    print("=" * 60)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} resumes")
        
        resume_col = 'Resume_str' if 'Resume_str' in df.columns else 'Resume'
        df = df.dropna(subset=[resume_col, 'Category'])
        
        preprocessor = ResumePreprocessor()
        
        print("Cleaning and enhancing text data...")
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        df['enhanced_features'] = df[resume_col].apply(preprocessor.extract_enhanced_features)
        df['enhanced_text'] = df['cleaned_text'] + ' ' + df['enhanced_features']
        
        # Filter short resumes
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 50]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        
        print(f"Final dataset: {len(df)} resumes")
        
        # FIRST: Analyze and fix class imbalance using Category column
        df = analyze_and_fix_class_imbalance(df)
        
        # THEN: Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        
        print(f"Class Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.head(10).items():
            print(f"  {category:25s}: {count:3d} samples")
        
        return df, label_map, num_labels
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets"""
    texts = df['enhanced_text'].tolist()
    labels = df['label'].tolist()
    categories = df['Category'].tolist()
    
    X_temp, X_test, y_temp, y_test, cat_temp, cat_test = train_test_split(
        texts, labels, categories,
        test_size=test_size,
        random_state=random_state, 
        stratify=categories
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, cat_train, cat_val = train_test_split(
        X_temp, y_temp, cat_temp,
        test_size=val_ratio,
        random_state=random_state, 
        stratify=cat_temp
    )
    
    print(f"Data Split Complete:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
