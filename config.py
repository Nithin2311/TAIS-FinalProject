"""
Configuration for resume classification system.
"""

class Config:
    """Configuration for resume classification"""
    
    # Model Settings
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    # Training Parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 3e-5
    WARMUP_RATIO = 0.15
    WEIGHT_DECAY = 0.015
    EARLY_STOPPING_PATIENCE = 4
    
    # Regularization
    DROPOUT_RATE = 0.25
    MAX_GRAD_NORM = 1.0
    
    # Data Split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = 'data/raw/Resume.csv'
    BASELINE_MODEL_PATH = 'models/resume_classifier_baseline'
    DEBIASED_MODEL_PATH = 'models/resume_classifier_debiased'
    
    # Bias Mitigation Settings
    USE_FOCAL_LOSS = True
    ENHANCED_BALANCING = True
    ADVERSARIAL_LAMBDA = 0.03
    MAX_SAMPLES_PER_GROUP = 100
    
    # Target categories for focused improvement
    TARGET_AUGMENTATION_CATEGORIES = ['BPO', 'AUTOMOBILE', 'APPAREL', 'DIGITAL-MEDIA', 'ARTS']
    
    # Explainability
    LIME_NUM_FEATURES = 10
    ENABLE_SHAP = True
    
    # Advanced Debiasing
    COUNTERFACTUAL_AUGMENTATION_FACTOR = 2.0
    MIN_SAMPLES_PER_CATEGORY = 20
    INTERSECTIONAL_DEBIASING_ENABLED = True
