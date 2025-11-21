"""
Enhanced Configuration settings for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class Config:
    """Optimized configuration for enhanced resume classification"""
    
    # Model Configuration
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    # Enhanced Training Parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 12  # Reduced to prevent overfitting
    LEARNING_RATE = 2e-5  # Slightly higher for better convergence
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3
    
    # Enhanced Regularization
    DROPOUT_RATE = 0.2
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_GRAD_NORM = 1.0
    
    # Data Configuration
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = 'data/raw/Resume.csv'
    MODEL_SAVE_PATH = 'models/resume_classifier'
    DEBIASED_MODEL_PATH = 'models/resume_classifier_debiased'
    
    # Enhanced Features
    USE_FOCAL_LOSS = True
    ENHANCED_BALANCING = True
    
    # Debiasing Parameters
    ADVERSARIAL_LAMBDA = 0.05  # Reduced for stability
    MAX_SAMPLES_PER_GROUP = 80  # Prevent dataset explosion
    TARGET_AUGMENTATION_CATEGORIES = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA']
    
    # Explainability
    LIME_NUM_FEATURES = 8
    ENABLE_SHAP = False  # Set to True if you want SHAP (memory intensive)
    
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
