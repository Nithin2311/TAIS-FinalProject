"""
Optimized Enhanced Data Processing Module for Resume Classification System
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
import nlpaug.augmenter.word as naw


class ResumePreprocessor:
    """Enhanced text preprocessing for resumes with demographic signal removal"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s\.,;:!?\-]')
        
        # Enhanced demographic indicators to remove
        self.gender_indicators = [
            'he', 'she', 'him', 'her', 'his', 'hers',
            'mr', 'mrs', 'ms', 'miss', 'sir', 'madam',
            'father', 'mother', 'husband', 'wife',
            'brother', 'sister', 'son', 'daughter',
            'male', 'female', 'man', 'woman', 'men', 'women',
            'boy', 'girl', 'gentleman', 'lady'
        ]
        
        self.privilege_indicators = [
            'harvard', 'stanford', 'mit', 'princeton', 'yale', 'columbia',
            'oxford', 'cambridge', 'ivy league', 'goldman sachs',
            'mckinsey', 'bain', 'boston consulting', 'google',
            'microsoft', 'apple', 'amazon', 'meta', 'facebook'
        ]
        
        # Names for removal
        self.common_names = [
            'john', 'jane', 'michael', 'sarah', 'david', 'emily',
            'james', 'jennifer', 'robert', 'lisa', 'william', 'mary',
            'richard', 'patricia', 'joseph', 'linda', 'thomas', 'barbara'
        ]

    def clean_text(self, text):
        """Clean text with enhanced demographic signal removal"""
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
        
        # Remove common names
        for name in self.common_names:
            text = re.sub(r'\b' + name + r'\b', ' ', text)
        
        # Remove special characters
        text = self.special_chars.sub(' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_enhanced_features(self, text):
        """Extract job-related features for improved classification"""
        text_lower = text.lower()
        
        technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'swift', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'cassandra'],
            'devops': ['ci/cd', 'jenkins', 'gitlab', 'github actions', 'monitoring', 'logging'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'xcode'],
            'business': ['project management', 'agile', 'scrum', 'stakeholder', 'strategy']
        }
        
        features = []
        
        for category, skills in technical_skills.items():
            found_skills = [skill for skill in skills if skill in text_lower]
            if found_skills:
                features.extend(found_skills)
        
        # Add experience level detection
        experience_patterns = [
            r'(\d+)\+?\s+years',
            r'(\d+)\s*-\s*(\d+)\s+years',
            r'senior\s+\w+',
            r'junior\s+\w+',
            r'entry[\s-]level',
            r'mid[\s-]level',
            r'experienced\s+\w+'
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


class DataAugmenter:
    """Enhanced data augmentation for underperforming categories"""
    
    def __init__(self):
        try:
            # Use synonym augmentation
            self.aug = naw.SynonymAug(aug_src='wordnet')
        except:
            self.aug = None
    
    def augment_text(self, text, n=3):
        """Generate augmented versions of text"""
        if self.aug is None:
            # Fallback simple augmentation
            words = text.split()
            augmented_texts = []
            for _ in range(n):
                np.random.shuffle(words[:min(20, len(words))])
                augmented_texts.append(' '.join(words))
            return augmented_texts
        
        try:
            return [self.aug.augment(text) for _ in range(n)]
        except:
            return [text] * n


def enhanced_balance_dataset(df, target_col='Category'):
    """Enhanced dataset balancing with targeted augmentation"""
    print("Applying enhanced dataset balancing with targeted augmentation...")
    
    from config import Config
    augmenter = DataAugmenter()
    
    target_categories = Config.TARGET_AUGMENTATION_CATEGORIES
    
    # Get category distribution
    category_counts = df[target_col].value_counts()
    max_count = category_counts.max()
    min_samples = Config.MIN_SAMPLES_PER_CATEGORY
    
    balanced_dfs = []
    
    for category in df[target_col].unique():
        category_df = df[df[target_col] == category].copy()
        current_count = len(category_df)
        
        # Determine target count
        if category in target_categories:
            target_count = max(max_count, min_samples * 3)
        else:
            target_count = max_count
        
        if current_count < target_count:
            needed = target_count - current_count
            
            # For target categories, use augmentation
            if category in target_categories and needed > 0:
                augmented_samples = []
                for idx in category_df.index[:min(10, len(category_df))]:
                    text = df.loc[idx, 'Resume_str' if 'Resume_str' in df.columns else 'Resume']
                    augmented_texts = augmenter.augment_text(text, n=min(5, needed // 10 + 1))
                    
                    for aug_text in augmented_texts:
                        if len(augmented_samples) >= needed:
                            break
                        new_sample = category_df.loc[[idx]].copy()
                        new_sample['Resume_str' if 'Resume_str' in new_sample.columns else 'Resume'] = aug_text
                        augmented_samples.append(new_sample)
                
                if augmented_samples:
                    augmented_df = pd.concat(augmented_samples, ignore_index=True)
                    category_df = pd.concat([category_df, augmented_df], ignore_index=True)
            
            # If still needed, duplicate
            if len(category_df) < target_count:
                additional_needed = target_count - len(category_df)
                duplicated = category_df.sample(n=additional_needed, replace=True, random_state=42)
                category_df = pd.concat([category_df, duplicated], ignore_index=True)
        
        balanced_dfs.append(category_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"Dataset balancing complete: {len(df)} -> {len(balanced_df)} samples")
    print(f"New distribution: {balanced_df[target_col].value_counts().to_dict()}")
    
    return balanced_df


def analyze_and_fix_class_imbalance(df, category_col='Category'):
    """Enhanced class imbalance analysis and fixing"""
    from config import Config
    
    category_counts = df[category_col].value_counts()
    
    print("Enhanced Class Imbalance Analysis")
    print("=" * 60)
    
    # Identify problematic categories
    problem_classes = category_counts[category_counts < Config.MIN_SAMPLES_PER_CATEGORY].index.tolist()
    low_performance_classes = Config.TARGET_AUGMENTATION_CATEGORIES
    
    all_problem_classes = list(set(problem_classes + low_performance_classes))
    
    if all_problem_classes:
        print(f"Identified {len(all_problem_classes)} classes for enhancement:")
        for cls in all_problem_classes:
            count = category_counts.get(cls, 0)
            print(f"  {cls}: {count} samples")
        
        df_balanced = enhanced_balance_dataset(df, category_col)
        return df_balanced
    else:
        print("No critical class imbalance detected")
        return df


def load_and_preprocess_data(data_path):
    """Load and preprocess resume data with enhanced processing"""
    print("Enhanced Data Processing with Advanced Features")
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
        df['combined_text'] = df['cleaned_text'] + ' ' + df['enhanced_features']
        
        # Filter short resumes
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 100]  # Increased from 50
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        
        print(f"Final dataset: {len(df)} resumes")
        
        # Enhanced class imbalance handling
        df = analyze_and_fix_class_imbalance(df)
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {str(i): cat for i, cat in enumerate(label_encoder.classes_)}  # Convert keys to strings
        num_labels = len(label_map)
        
        print(f"Enhanced Class Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category:25s}: {count:3d} samples")
        
        # Calculate and print statistics
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
    
    # Ensure stratification works for all categories
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
