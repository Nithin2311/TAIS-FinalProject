"""
Enhanced Data Processing for Resume Classification
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

import re
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

class EnhancedResumePreprocessor:
    """Advanced text preprocessing with semantic preservation"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        
        # Enhanced technical skills dictionary
        self.technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node', 'express', 'spring'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd'],
            'ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'nlp', 'computer vision'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'vscode', 'pycharm', 'eclipse']
        }
        
        # Resume-specific important terms (DO NOT remove these)
        self.important_terms = {
            'experience', 'education', 'skills', 'projects', 'certifications',
            'bachelor', 'master', 'phd', 'degree', 'university', 'college',
            'engineer', 'developer', 'analyst', 'manager', 'director', 'specialist',
            'software', 'data', 'business', 'financial', 'marketing', 'sales'
        }
        
        self.stop_words = set(stopwords.words('english')) - self.important_terms
    
    def clean_text(self, text):
        """Enhanced text cleaning that preserves semantic meaning"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' [URL] ', text)
        text = self.email_pattern.sub(' [EMAIL] ', text)
        text = self.phone_pattern.sub(' [PHONE] ', text)
        
        # Preserve case for important terms, lowercase the rest
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Keep important terms in original case
            if word.lower() in self.important_terms:
                cleaned_words.append(word)
            else:
                # Clean and lowercase other words
                cleaned_word = re.sub(r'[^a-zA-Z0-9\s]', '', word.lower())
                if cleaned_word and len(cleaned_word) > 1 and cleaned_word not in self.stop_words:
                    cleaned_words.append(cleaned_word)
        
        # Smart length normalization - preserve key sections
        if len(cleaned_words) > 800:
            # Keep first 500 words (usually experience/skills) and last 300 (education/summary)
            cleaned_words = cleaned_words[:500] + cleaned_words[-300:]
        
        return ' '.join(cleaned_words)
    
    def extract_enhanced_features(self, text):
        """Extract structured features from resume text"""
        features = {}
        text_lower = text.lower()
        
        # Extract technical skills with categories
        for category, skills in self.technical_skills.items():
            found_skills = [skill for skill in skills if skill in text_lower]
            if found_skills:
                features[f'{category}_skills'] = ' '.join(found_skills)
        
        # Extract education level
        education_indicators = {
            'phd': ['phd', 'doctorate', 'doctoral'],
            'masters': ['master', 'ms', 'm.s.', 'mba', 'm.sc'],
            'bachelor': ['bachelor', 'bs', 'b.s.', 'ba', 'b.a', 'undergraduate'],
            'associate': ['associate', 'a.a', 'a.s']
        }
        
        for level, indicators in education_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                features['education_level'] = level
                break
        
        # Extract experience indicators
        experience_pattern = r'(\d+)\+?\s*years?'
        experience_matches = re.findall(experience_pattern, text)
        if experience_matches:
            features['experience_years'] = max(map(int, experience_matches))
        
        return features
    
    def enhance_resume_text(self, text):
        """Combine cleaned text with extracted features"""
        cleaned_text = self.clean_text(text)
        features = self.extract_enhanced_features(text)
        
        # Append features to text
        enhanced_text = cleaned_text
        for feature_value in features.values():
            if isinstance(feature_value, str):
                enhanced_text += ' ' + feature_value
            else:
                enhanced_text += ' ' + str(feature_value)
        
        return enhanced_text.strip()

def load_and_enhance_data(data_path):
    """Load and preprocess data with enhanced features"""
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
        preprocessor = EnhancedResumePreprocessor()
        
        # Enhanced preprocessing
        print("Applying enhanced text preprocessing...")
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        df['enhanced_text'] = df[resume_col].apply(preprocessor.enhance_resume_text)
        
        # Filter out very short resumes
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 150]  # Increased minimum length
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        print(f"Enhanced {len(df)} resumes")
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        
        # Enhanced class distribution analysis
        print(f"\nEnhanced Class Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category:25s}: {count:3d} samples ({percentage:.1f}%)")
        
        # Identify imbalanced classes
        min_samples = category_counts.min()
        max_samples = category_counts.max()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        print(f"\nClass imbalance: {min_samples} to {max_samples} (ratio: {imbalance_ratio:.1f}x)")
        
        # Analyze text length distribution
        text_lengths = df['enhanced_text'].str.len()
        print(f"\nText Length Statistics:")
        print(f"  Average length: {text_lengths.mean():.0f} characters")
        print(f"  Min length: {text_lengths.min()} characters")
        print(f"  Max length: {text_lengths.max()} characters")
        
        return df, label_map, num_labels
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_balanced_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """Create balanced splits addressing class imbalance"""
    texts = df['enhanced_text'].tolist()
    labels = df['label'].tolist()
    categories = df['Category'].tolist()
    
    # Use stratification to maintain class distribution
    X_temp, X_test, y_temp, y_test, cat_temp, cat_test = train_test_split(
        texts, labels, categories, 
        test_size=test_size,
        random_state=random_state, 
        stratify=categories
    )
    
    # Adjust validation split
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
    
    # Verify class distribution in splits
    def get_class_distribution(labels, label_map):
        distribution = {}
        total = len(labels)
        for label in set(labels):
            count = sum(1 for l in labels if l == label)
            distribution[label_map[label]] = f"{count} ({count/total*100:.1f}%)"
        return distribution
    
    print(f"\nClass Distribution in Splits:")
    print(f"  Train: {get_class_distribution(y_train, label_map)}")
    print(f"  Val:   {get_class_distribution(y_val, label_map)}")
    print(f"  Test:  {get_class_distribution(y_test, label_map)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
