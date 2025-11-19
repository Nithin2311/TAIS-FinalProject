"""
Enhanced Data loading and preprocessing module
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
import json
from collections import Counter


class ResumePreprocessor:
    """Enhanced text preprocessing pipeline for resumes"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s\.,;:!?\-]')
        
        # Keep important section headers for context
        self.section_headers = {
            'experience', 'education', 'skills', 'projects', 'certifications',
            'summary', 'objective', 'work', 'employment', 'technical'
        }
    
    def clean_text(self, text):
        """Enhanced text cleaning that preserves important context"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase but preserve section headers
        text = text.lower()
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove special characters but keep basic punctuation
        text = self.special_chars.sub(' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Enhanced length normalization - keep most informative sections
        # Instead of arbitrary truncation, we'll use the full context up to max length
        # The tokenizer will handle truncation properly
        
        return text.strip()
    
    def extract_enhanced_features(self, text):
        """Extract enhanced features to improve classification"""
        text_lower = text.lower()
        
        # Technical skills patterns
        technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node.js'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis']
        }
        
        # Extract features
        features = []
        for category, skills in technical_skills.items():
            found_skills = [skill for skill in skills if skill in text_lower]
            if found_skills:
                features.extend(found_skills)
        
        return ' '.join(features)
    
    def detect_resume_sections(self, text):
        """Detect and weight important resume sections"""
        sections = {}
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            if any(header in line_clean for header in self.section_headers):
                # Found a section header, include more content from this section
                section_content = []
                for j in range(i, min(i + 10, len(lines))):
                    section_content.append(lines[j])
                sections[line_clean] = ' '.join(section_content)
        
        return sections


def download_dataset():
    """Download dataset from Google Drive if not exists"""
    os.makedirs('data/raw', exist_ok=True)
    
    if not os.path.exists('data/raw/Resume.csv'):
        print("Downloading dataset from Google Drive...")
        try:
            gdown.download(
                'https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk',
                'data/raw/Resume.csv', 
                quiet=False
            )
            print("Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    else:
        print("Dataset already exists!")
        return True


def analyze_class_distribution(df, category_col='Category'):
    """Comprehensive analysis of class distribution"""
    category_counts = df[category_col].value_counts()
    
    print(f"\nCOMPREHENSIVE CLASS DISTRIBUTION ANALYSIS:")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Number of categories: {len(category_counts)}")
    print(f"Min samples per class: {category_counts.min()}")
    print(f"Max samples per class: {category_counts.max()}")
    print(f"Imbalance ratio: {category_counts.max()/category_counts.min():.2f}x")
    
    # Identify problematic classes
    problem_classes = category_counts[category_counts < 10]
    if len(problem_classes) > 0:
        print(f"\nWARNING: {len(problem_classes)} classes have < 10 samples:")
        for cls, count in problem_classes.items():
            print(f"  {cls}: {count} samples")


def load_and_preprocess_data(data_path):
    """Enhanced data loading and preprocessing"""
    print("=" * 60)
    print("ENHANCED DATA PROCESSING")
    print("=" * 60)
    
    try:
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} resumes")
        
        # Identify resume column
        resume_col = 'Resume_str' if 'Resume_str' in df.columns else 'Resume'
        print(f"Using column: {resume_col}")
        
        # Clean data
        df = df.dropna(subset=[resume_col, 'Category'])
        preprocessor = ResumePreprocessor()
        
        # Enhanced cleaning with feature extraction
        print("Cleaning and enhancing text data...")
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        df['enhanced_features'] = df[resume_col].apply(preprocessor.extract_enhanced_features)
        
        # Combine cleaned text with enhanced features
        df['enhanced_text'] = df['cleaned_text'] + ' ' + df['enhanced_features']
        
        # Filter out very short resumes
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 50]  # Reasonable minimum
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        print(f"Final dataset: {len(df)} resumes")
        
        # Analyze class distribution
        analyze_class_distribution(df)
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        
        print(f"\nClass Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category:25s}: {count:3d} samples")
        
        return df, label_map, num_labels
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Enhanced data splitting with stratification and balance checking"""
    texts = df['enhanced_text'].tolist()
    labels = df['label'].tolist()
    categories = df['Category'].tolist()
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test, cat_temp, cat_test = train_test_split(
        texts, labels, categories,
        test_size=test_size,
        random_state=random_state, 
        stratify=categories
    )
    
    # Adjust validation split to get exactly 15% of total
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, cat_train, cat_val = train_test_split(
        X_temp, y_temp, cat_temp,
        test_size=val_ratio,
        random_state=random_state, 
        stratify=cat_temp
    )
    
    print(f"\nEnhanced Data Split Complete:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
    
    # Verify stratification
    print(f"\nClass distribution in splits:")
    for split_name, split_cats in [("Train", cat_train), ("Val", cat_val), ("Test", cat_test)]:
        cat_counts = Counter(split_cats)
        print(f"  {split_name}: {len(cat_counts)} classes, samples: {len(split_cats)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
