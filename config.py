"""
Configuration settings for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class Config:
    """Configuration for resume classification"""
    
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 12
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3
    
    DROPOUT_RATE = 0.2
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_GRAD_NORM = 1.0
    
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    DATA_PATH = 'data/raw/Resume.csv'
    BASELINE_MODEL_PATH = 'models/resume_classifier_baseline'
    DEBIASED_MODEL_PATH = 'models/resume_classifier_debiased'
    
    USE_FOCAL_LOSS = True
    ENHANCED_BALANCING = True
    
    ADVERSARIAL_LAMBDA = 0.05
    MAX_SAMPLES_PER_GROUP = 80
    TARGET_AUGMENTATION_CATEGORIES = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA']
    
    LIME_NUM_FEATURES = 8
    ENABLE_SHAP = False
    
    @staticmethod
    def display_enhanced_config():
        """Display enhanced configuration for final submission"""
        print("=" * 60)
        print("ENHANCED PROJECT CONFIGURATION - FINAL SUBMISSION")
        print("=" * 60)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Max Length: {Config.MAX_LENGTH} tokens")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Focal Loss: {Config.USE_FOCAL_LOSS}")
        print(f"Enhanced Balancing: {Config.ENHANCED_BALANCING}")
        print(f"Dropout: {Config.DROPOUT_RATE}")
        print(f"Early Stopping: {Config.EARLY_STOPPING_PATIENCE}")
        print(f"Adversarial Lambda: {Config.ADVERSARIAL_LAMBDA}")
        print(f"Target Augmentation Categories: {Config.TARGET_AUGMENTATION_CATEGORIES}")
        print("=" * 60)
