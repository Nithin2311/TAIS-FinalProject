"""
Data processing module for resume classification system.
"""

import re
import os
import pandas as pd
import numpy as np
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from collections import Counter


class ResumePreprocessor:
    """Text preprocessing for resumes"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s\.,;:!?\-]')
    
    def clean_text(self, text):
        """Clean text for model input"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        text = self.special_chars.sub(' ', text)
        
        text = ' '.join(text.split())
        return text.strip()
    
    def extract_features(self, text):
        """Extract job-related features for improved classification"""
        text_lower = text.lower()
        
        technical_skills = {
            'programming': [
                r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\bc\+\+\b', r'\bc#\b', 
                r'\bruby\b', r'\bgo\b', r'\bswift\b', r'\bkotlin\b', r'\btypescript\b',
                r'\bphp\b', r'\bperl\b', r'\br\b', r'\bscala\b', r'\bhaskell\b'
            ],
            'web': [
                r'\bhtml\b', r'\bcss\b', r'\breact\b', r'\bangular\b', r'\bvue\b', 
                r'\bnode\.js\b', r'\bnodejs\b', r'\bdjango\b', r'\bflask\b', r'\bspring\b',
                r'\bexpress\b', r'\bfastapi\b'
            ],
            'data_science': [
                r'\bmachine learning\b', r'\bdeep learning\b', r'\btensorflow\b', 
                r'\bpytorch\b', r'\bkeras\b', r'\bscikit-learn\b', r'\bxgboost\b',
                r'\bpandas\b', r'\bnumpy\b', r'\bscipy\b', r'\bjupyter\b'
            ],
            'cloud': [
                r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bgoogle cloud\b', r'\bdocker\b',
                r'\bkubernetes\b', r'\bterraform\b', r'\bansible\b'
            ],
            'database': [
                r'\bsql\b', r'\bmysql\b', r'\bpostgresql\b', r'\bmongodb\b', r'\bredis\b',
                r'\boracle\b', r'\bcassandra\b', r'\bsqlite\b'
            ],
            'devops': [
                r'\bci/cd\b', r'\bjenkins\b', r'\bgitlab\b', r'\bgithub actions\b',
                r'\bmonitoring\b', r'\blogging\b'
            ],
            'mobile': [
                r'\bandroid\b', r'\bios\b', r'\breact native\b', r'\bflutter\b', r'\bxcode\b'
            ],
            'business': [
                r'\bproject management\b', r'\bagile\b', r'\bscrum\b', r'\bstakeholder\b',
                r'\bstrategy\b'
            ]
        }
        
        features = []
        
        for category, patterns in technical_skills.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Extract just the skill name without word boundaries
                    skill = re.sub(r'^\\b|\\b$', '', pattern)
                    skill = re.sub(r'\\\\', '', skill)  # Remove escape characters
                    features.append(skill)
        
        # Experience detection - more robust
        experience_patterns = [
            r'(\d+)\+?\s+years?\s+experience',
            r'(\d+)\s*-\s*(\d+)\s+years?\s+experience',
            r'experienced\s+\w+',
            r'senior\s+\w+',
            r'junior\s+\w+',
            r'entry[\s-]level',
            r'mid[\s-]level'
        ]
        
        for pattern in experience_patterns:
            if re.search(pattern, text_lower):
                features.append('experience_mentioned')
                break
        
        return ' '.join(features)


def download_dataset():
    """Download dataset from Google Drive"""
    os.makedirs('data/raw', exist_ok=True)
    
    if not os.path.exists('data/raw/Resume.csv'):
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


def balance_dataset(df, target_col='Category'):
    """Apply dataset balancing for underperforming categories"""
    from config import Config
    
    target_categories = Config.TARGET_AUGMENTATION_CATEGORIES
    min_samples = Config.MIN_SAMPLES_PER_CATEGORY
    
    category_counts = df[target_col].value_counts()
    max_count = category_counts.max()
    
    balanced_dfs = []
    
    for category in df[target_col].unique():
        category_df = df[df[target_col] == category].copy()
        current_count = len(category_df)
        
        if category in target_categories:
            target_count = max(max_count, min_samples * 3)
        else:
            target_count = max_count
        
        if current_count < target_count:
            needed = target_count - current_count
            
            if needed > 0:
                duplicated = category_df.sample(n=needed, replace=True, random_state=42)
                category_df = pd.concat([category_df, duplicated], ignore_index=True)
        
        balanced_dfs.append(category_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"Dataset balancing complete: {len(df)} -> {len(balanced_df)} samples")
    print(f"New distribution: {balanced_df[target_col].value_counts().to_dict()}")
    
    return balanced_df


def analyze_and_fix_class_imbalance(df, category_col='Category'):
    """Class imbalance analysis and fixing"""
    from config import Config
    
    category_counts = df[category_col].value_counts()
    
    print("Class Imbalance Analysis")
    print("=" * 60)
    
    problem_classes = category_counts[category_counts < Config.MIN_SAMPLES_PER_CATEGORY].index.tolist()
    low_performance_classes = Config.TARGET_AUGMENTATION_CATEGORIES
    
    all_problem_classes = list(set(problem_classes + low_performance_classes))
    
    if all_problem_classes:
        print(f"Identified {len(all_problem_classes)} classes for enhancement:")
        for cls in all_problem_classes:
            count = category_counts.get(cls, 0)
            print(f"  {cls}: {count} samples")
        
        df_balanced = balance_dataset(df, category_col)
        return df_balanced
    else:
        print("No critical class imbalance detected")
        return df


def load_and_preprocess_data(data_path):
    """Load and preprocess resume data"""
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} resumes")
        
        resume_col = 'Resume_str' if 'Resume_str' in df.columns else 'Resume'
        df = df.dropna(subset=[resume_col, 'Category'])
        
        preprocessor = ResumePreprocessor()
        
        print("Cleaning text data...")
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        df['enhanced_features'] = df[resume_col].apply(preprocessor.extract_features)
        df['combined_text'] = df['cleaned_text'] + ' ' + df['enhanced_features']
        
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 100]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        
        print(f"Final dataset: {len(df)} resumes")
        
        df = analyze_and_fix_class_imbalance(df)
        
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {str(i): cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        
        print("Class Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category:25s}: {count:3d} samples")
        
        print(f"\nStatistics:")
        print(f"  Average text length: {df['cleaned_text'].str.len().mean():.0f} characters")
        print(f"  Number of categories: {num_labels}")
        
        return df, label_map, num_labels
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets with stratification"""
    texts = df['combined_text'].tolist()
    labels = df['label'].tolist()
    categories = df['Category'].tolist()
    
    min_samples_for_stratify = 2
    category_counts = Counter(categories)
    valid_categories = [cat for cat, count in category_counts.items() if count >= min_samples_for_stratify]
    
    if len(valid_categories) < len(category_counts):
        print(f"Warning: Some categories have fewer than {min_samples_for_stratify} samples")
    
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
