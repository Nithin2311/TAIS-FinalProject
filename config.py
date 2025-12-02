"""
Enhanced Configuration with Optimized Hyperparameters
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class Config:
    """Optimized configuration for resume classification"""
    
    # Model Settings
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    # Training Parameters (Optimized)
    BATCH_SIZE = 16  # Increased from 8 for better gradient estimation
    NUM_EPOCHS = 15  # Increased from 12 for better convergence
    LEARNING_RATE = 3e-5  # Adjusted for finer tuning
    WARMUP_RATIO = 0.15
    WEIGHT_DECAY = 0.015
    EARLY_STOPPING_PATIENCE = 4  # Increased for stability
    
    # Regularization
    DROPOUT_RATE = 0.25  # Increased to prevent overfitting
    GRADIENT_ACCUMULATION_STEPS = 1  # Removed for simplicity
    MAX_GRAD_NORM = 1.0
    
    # Data Split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = 'data/raw/Resume.csv'
    BASELINE_MODEL_PATH = 'models/resume_classifier_baseline'
    DEBIASED_MODEL_PATH = 'models/resume_classifier_debiased'
    ADVANCED_DEBIASED_MODEL_PATH = 'models/resume_classifier_advanced_debiased'
    
    # Bias Mitigation Settings (Optimized)
    USE_FOCAL_LOSS = True
    ENHANCED_BALANCING = True
    ADVERSARIAL_LAMBDA = 0.03  # Adjusted for better balance
    MAX_SAMPLES_PER_GROUP = 100  # Increased for better representation
    
    # Target categories for focused improvement
    TARGET_AUGMENTATION_CATEGORIES = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA', 'ARTS']
    
    # Explainability
    LIME_NUM_FEATURES = 10
    ENABLE_SHAP = True
    
    # Advanced Debiasing
    COUNTERFACTUAL_AUGMENTATION_FACTOR = 2.0
    MIN_SAMPLES_PER_CATEGORY = 20
    INTERSECTIONAL_DEBIASING_ENABLED = True
    
    @staticmethod
    def display_enhanced_config():
        """Display enhanced configuration for final submission"""
        print("=" * 60)
        print("OPTIMIZED PROJECT CONFIGURATION - FINAL SUBMISSION")
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
        print(f"Target Categories: {Config.TARGET_AUGMENTATION_CATEGORIES}")
        print("=" * 60)
