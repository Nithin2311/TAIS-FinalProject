"""
Enhanced Configuration for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class EnhancedConfig:
    """Optimized configuration for enhanced resume classification"""
    
    # Model Configuration
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    # Enhanced Training Parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 25  # Reduced for better generalization
    LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 5
    
    # Enhanced Data Configuration
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Enhanced Paths
    DATA_PATH = 'data/raw/Resume.csv'
    MODEL_SAVE_PATH = 'models/enhanced_resume_classifier'
    GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk'
    
    # Enhanced Training Features
    USE_CLASS_WEIGHTS = True
    LABEL_SMOOTHING = 0.1
    GRADIENT_ACCUMULATION_STEPS = 1
    
    @staticmethod
    def display_config():
        """Display enhanced configuration"""
        print("=" * 60)
        print("ENHANCED PROJECT CONFIGURATION")
        print("=" * 60)
        print(f"Model: {EnhancedConfig.MODEL_NAME}")
        print(f"Max Length: {EnhancedConfig.MAX_LENGTH} tokens")
        print(f"Batch Size: {EnhancedConfig.BATCH_SIZE}")
        print(f"Epochs: {EnhancedConfig.NUM_EPOCHS}")
        print(f"Learning Rate: {EnhancedConfig.LEARNING_RATE}")
        print(f"Use Class Weights: {EnhancedConfig.USE_CLASS_WEIGHTS}")
        print(f"Label Smoothing: {EnhancedConfig.LABEL_SMOOTHING}")
        print(f"Train/Val/Test Split: 70%/15%/15%")
        print("=" * 60)
