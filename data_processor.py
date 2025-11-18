"""
Data loading and preprocessing module
CAI 6605 - Trustworthy AI Systems - Midterm Project
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
    """Text preprocessing pipeline for resumes"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
    
    def clean_text(self, text):
        """Clean and normalize resume text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\-\']', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Smart length normalization (keep important sections)
        words = text.split()
        if len(words) > 800:
            # Keep first 600 words (experience/skills) and last 200 (education/summary)
            text = ' '.join(words[:600] + words[-200:])
        
        return text.strip()


def download_dataset():
    """Download dataset from Google Drive if not exists"""
    os.makedirs('data/raw', exist_ok=True)
    
    if not os.path.exists('data/raw/Resume.csv'):
        print("📥 Downloading dataset from Google Drive...")
        try:
            gdown.download(
                'https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk',
                'data/raw/Resume.csv', 
                quiet=False
            )
            print("✅ Dataset downloaded successfully!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    else:
        print("✅ Dataset already exists!")
    
    return True


def load_and_preprocess_data(data_path):
    """Load and preprocess the resume dataset"""
    print("=" * 70)
    print("📊 PROCESSING DATA")
    print("=" * 70)
    
    try:
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} resumes")
        
        # Identify resume column
        resume_col = 'Resume_str' if 'Resume_str' in df.columns else 'Resume'
        print(f"✅ Using column: {resume_col}")
        
        # Clean data
        df = df.dropna(subset=[resume_col, 'Category'])
        preprocessor = ResumePreprocessor()
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        
        # Filter out very short resumes
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 50]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"✅ Filtered {filtered_count} resumes with insufficient text")
        print(f"✅ Cleaned {len(df)} resumes")
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        print(f"✅ {num_labels} job categories found")
        
        # Display categories
        print("\n📋 Job Categories:")
        for idx, category in enumerate(label_encoder.classes_):
            print(f"  {idx+1:2d}. {category}")
        
        return df, label_map, num_labels
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None, None, None


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split data into train, validation, and test sets"""
    texts = df['cleaned_text'].tolist()
    labels = df['label'].tolist()
    
    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, 
        test_size=test_size,
        random_state=random_state, 
        stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=0.176,  # 0.176 of 85% = 15% of total
        random_state=random_state, 
        stratify=y_temp
    )
    
    print(f"\n✅ Data Split Complete:")
    print(f"  • Train: {len(X_train)} samples (70%)")
    print(f"  • Val:   {len(X_val)} samples (15%)")
    print(f"  • Test:  {len(X_test)} samples (15%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
