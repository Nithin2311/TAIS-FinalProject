"""
Configuration settings for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class Config:
    """Optimized configuration for resume classification"""
    
    # Model Configuration
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    # Training Parameters - OPTIMIZED
    BATCH_SIZE = 8  # Reduced for better stability
    NUM_EPOCHS = 15  # Reduced to prevent overfitting
    LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3
    
    # Data Configuration
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = 'data/raw/Resume.csv'
    MODEL_SAVE_PATH = 'models/resume_classifier'
    GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk'
    
    # Enhanced Training
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size of 16
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    
    @staticmethod
    def display_config():
        """Display configuration for presentation"""
        print("=" * 60)
        print("OPTIMIZED PROJECT CONFIGURATION")
        print("=" * 60)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Max Length: {Config.MAX_LENGTH} tokens")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Effective Batch Size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Early Stopping: {Config.EARLY_STOPPING_PATIENCE}")
        print(f"Train/Val/Test Split: 70%/15%/15%")
        print("=" * 60)
