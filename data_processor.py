"""
Data loading and preprocessing module
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


class ResumePreprocessor:
    """Advanced text preprocessing pipeline for resumes"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s\.,;:!?\-]')
        
        # Domain-specific stop words that don't carry semantic meaning
        self.resume_stop_words = {
            'objective', 'summary', 'experience', 'education', 'skills', 
            'projects', 'certifications', 'references', 'work', 'history',
            'phone', 'email', 'linkedin', 'github', 'http', 'https', 'www'
        }
    
    def clean_text(self, text):
        """Comprehensive text cleaning and normalization"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove special characters but keep basic punctuation
        text = self.special_chars.sub(' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove resume-specific stop words
        words = text.split()
        filtered_words = [word for word in words if word not in self.resume_stop_words]
        text = ' '.join(filtered_words)
        
        # Enhanced length normalization - keep most informative sections
        words = text.split()
        if len(words) > 1000:
            # Keep first 700 words (experience/skills) and last 300 (education/summary)
            text = ' '.join(words[:700] + words[-300:])
        elif len(words) > 600:
            text = ' '.join(words[:400] + words[-200:])
        
        return text.strip()
    
    def extract_technical_skills(self, text):
        """Extract technical skills to enhance feature representation"""
        technical_keywords = {
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 
            'docker', 'kubernetes', 'machine learning', 'deep learning',
            'tensorflow', 'pytorch', 'scikit', 'django', 'flask', 'fastapi',
            'mongodb', 'postgresql', 'mysql', 'git', 'jenkins', 'ci/cd',
            'azure', 'gcp', 'linux', 'rest', 'api', 'microservices'
        }
        
        found_skills = [skill for skill in technical_keywords if skill in text.lower()]
        return ' '.join(found_skills)


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


def load_and_preprocess_data(data_path):
    """Load and preprocess the resume dataset with enhanced cleaning"""
    print("=" * 60)
    print("DATA PROCESSING")
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
        
        # Enhanced cleaning with skill extraction
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        df['technical_skills'] = df[resume_col].apply(preprocessor.extract_technical_skills)
        
        # Combine cleaned text with technical skills for better representation
        df['enhanced_text'] = df['cleaned_text'] + ' ' + df['technical_skills']
        
        # Filter out very short resumes
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 100]  # Increased minimum length
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        print(f"Cleaned {len(df)} resumes")
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        
        # Analyze class distribution
        print(f"\nClass Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category:25s}: {count:3d} samples")
        
        # Identify imbalanced classes
        min_samples = category_counts.min()
        max_samples = category_counts.max()
        print(f"\nClass imbalance: {min_samples} to {max_samples} (ratio: {max_samples/min_samples:.1f}x)")
        
        return df, label_map, num_labels
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None, None


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets with stratification"""
    texts = df['enhanced_text'].tolist()  # Use enhanced text
    labels = df['label'].tolist()
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, 
        test_size=test_size,
        random_state=random_state, 
        stratify=labels
    )
    
    # Adjust validation split to get exactly 15% of total
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_ratio,
        random_state=random_state, 
        stratify=y_temp
    )
    
    print(f"\nData Split Complete:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
