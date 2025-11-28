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
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import json
from collections import Counter


class ResumePreprocessor:
    """Enhanced text preprocessing pipeline for resumes"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.special_chars = re.compile(r'[^a-zA-Z0-9\s\.,;:!?\-]')
        
        self.section_headers = {
            'experience', 'education', 'skills', 'projects', 'certifications',
            'summary', 'objective', 'work', 'employment', 'technical'
        }
        
        self.category_keywords = {
            'BPO': ['call center', 'customer service', 'bpo', 'voice process', 'non-voice',
                   'customer support', 'telecalling', 'inbound', 'outbound', 'helpdesk',
                   'client service', 'contact center', 'customer care', 'technical support'],
            
            'AUTOMOBILE': ['automobile', 'auto', 'vehicle', 'car', 'mechanic', 'automotive',
                          'engine repair', 'transmission', 'brake system', 'suspension',
                          'diagnostic', 'maintenance', 'service technician', 'auto repair',
                          'dealership', 'workshop', 'mechanical engineer', 'automobile engineer'],
            
            'APPAREL': ['apparel', 'fashion', 'clothing', 'garment', 'textile', 'merchandising',
                       'retail', 'fashion design', 'textile design', 'pattern making',
                       'garment manufacturing', 'fashion merchandiser', 'retail buyer',
                       'visual merchandising', 'fashion stylist', 'textile technology'],
            
            'DIGITAL-MEDIA': ['digital media', 'social media', 'content creation', 'video editing',
                             'graphic design', 'multimedia', 'animation', 'visual effects',
                             'digital marketing', 'content strategy', 'social media management',
                             'video production', 'motion graphics', 'ui/ux design', 'web design',
                             'digital content', 'media production']
        }
    
    def clean_text(self, text):
        """Enhanced text cleaning that preserves important context"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        text = self.special_chars.sub(' ', text)
        
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_enhanced_features(self, text):
        """Extract enhanced features to improve classification, especially for underperforming categories"""
        text_lower = text.lower()
        
        technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node.js', 'express'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'devops', 'terraform'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server']
        }
        
        features = []
        
        for category, skills in technical_skills.items():
            found_skills = [skill for skill in skills if skill in text_lower]
            if found_skills:
                features.extend(found_skills)
        
        for category, keywords in self.category_keywords.items():
            found_keywords = [f"CAT_{category}_{kw}" for kw in keywords if kw in text_lower]
            if found_keywords:
                if category in ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA']:
                    features.extend(found_keywords * 2)
                else:
                    features.extend(found_keywords)
        
        return ' '.join(features)
    
    def detect_resume_sections(self, text):
        """Detect and weight important resume sections"""
        sections = {}
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            if any(header in line_clean for header in self.section_headers):
                section_content = []
                for j in range(i, min(i + 10, len(lines))):
                    section_content.append(lines[j])
                sections[line_clean] = ' '.join(section_content)
        
        return sections
    
    def augment_text_for_category(self, text, target_category):
        """Augment text with category-specific keywords for data augmentation"""
        augmented_text = text
        
        if target_category in self.category_keywords:
            keywords = self.category_keywords[target_category][:3]
            augmented_text += " " + " ".join(keywords)
        
        return augmented_text


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


def enhanced_balance_dataset(df, target_col='Category'):
    """Enhanced dataset balancing with targeted augmentation for underperforming categories"""
    print("Applying enhanced dataset balancing with targeted augmentation...")
    
    underperforming_categories = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA']
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(df['enhanced_text'])
    y = df['label']
    
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    balanced_indices = []
    label_counts = Counter(y_balanced)
    preprocessor = ResumePreprocessor()
    
    for label in np.unique(y_balanced):
        label_indices = df[df['label'] == label].index.tolist()
        needed_count = label_counts[label]
        category_name = df[df['label'] == label]['Category'].iloc[0] if len(df[df['label'] == label]) > 0 else "Unknown"
        
        if len(label_indices) < needed_count:
            duplicates_needed = needed_count - len(label_indices)
            
            if category_name in underperforming_categories and duplicates_needed > 0:
                augmented_indices = []
                for i in range(duplicates_needed):
                    original_idx = np.random.choice(label_indices)
                    original_text = df.iloc[original_idx]['enhanced_text']
                    augmented_text = preprocessor.augment_text_for_category(original_text, category_name)
                    augmented_indices.append(original_idx)
                
                balanced_indices.extend(label_indices)
                balanced_indices.extend(augmented_indices)
            else:
                duplicated = np.random.choice(label_indices, duplicates_needed, replace=True)
                balanced_indices.extend(label_indices)
                balanced_indices.extend(duplicated.tolist())
        else:
            balanced_indices.extend(label_indices[:needed_count])
    
    balanced_df = df.iloc[balanced_indices].copy()
    balanced_df.reset_index(drop=True, inplace=True)
    
    for i in range(len(balanced_df)):
        category = balanced_df.iloc[i]['Category']
        if category in underperforming_categories and i >= len(df):
            original_text = balanced_df.iloc[i]['enhanced_text']
            augmented_text = preprocessor.augment_text_for_category(original_text, category)
            balanced_df.at[balanced_df.index[i], 'enhanced_text'] = augmented_text
    
    print(f"Enhanced dataset balancing complete: {len(df)} -> {len(balanced_df)} samples")
    
    print("Enhanced category distribution after balancing:")
    category_counts = balanced_df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category:25s}: {count:3d} samples")
    
    return balanced_df


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
    
    problem_classes = category_counts[category_counts < 10]
    if len(problem_classes) > 0:
        print(f"\nWARNING: {len(problem_classes)} classes have < 10 samples:")
        for cls, count in problem_classes.items():
            print(f"  {cls}: {count} samples")


def analyze_and_fix_class_imbalance(df, category_col='Category'):
    """Comprehensive class imbalance analysis and fixing"""
    category_counts = df[category_col].value_counts()
    
    print("\n" + "=" * 60)
    print("ENHANCED CLASS IMBALANCE ANALYSIS & FIXING")
    print("=" * 60)
    
    problem_classes = category_counts[category_counts < 10]
    if len(problem_classes) > 0:
        print(f"CRITICAL: {len(problem_classes)} classes have < 10 samples:")
        for cls, count in problem_classes.items():
            print(f"  {cls}: {count} samples")
        
        df_balanced = enhanced_balance_dataset(df)
        return df_balanced
    else:
        print("No critical class imbalance detected")
        return df


def load_and_preprocess_data(data_path):
    """Enhanced data loading and preprocessing"""
    print("=" * 60)
    print("ENHANCED DATA PROCESSING")
    print("=" * 60)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} resumes")
        
        resume_col = 'Resume_str' if 'Resume_str' in df.columns else 'Resume'
        print(f"Using column: {resume_col}")
        
        df = df.dropna(subset=[resume_col, 'Category'])
        preprocessor = ResumePreprocessor()
        
        print("Cleaning and enhancing text data...")
        df['cleaned_text'] = df[resume_col].apply(preprocessor.clean_text)
        df['enhanced_features'] = df[resume_col].apply(preprocessor.extract_enhanced_features)
        
        df['enhanced_text'] = df['cleaned_text'] + ' ' + df['enhanced_features']
        
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 50]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            print(f"Filtered {filtered_count} resumes with insufficient text")
        print(f"Final dataset: {len(df)} resumes")
        
        df = analyze_and_fix_class_imbalance(df)
        
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['Category'])
        label_map = {i: cat for i, cat in enumerate(label_encoder.classes_)}
        num_labels = len(label_map)
        
        print(f"\nEnhanced Class Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.head(15).items():
            status = "Good" if count >= 50 else "Medium" if count >= 20 else "Low"
            print(f"  {status} {category:25s}: {count:3d} samples")
        
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
    
    print(f"\nEnhanced Data Split Complete:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
    
    print(f"\nClass distribution in splits:")
    for split_name, split_cats in [("Train", cat_train), ("Val", cat_val), ("Test", cat_test)]:
        cat_counts = Counter(split_cats)
        print(f"  {split_name}: {len(cat_counts)} classes, samples: {len(split_cats)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
