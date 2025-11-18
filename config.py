"""
Configuration settings for Resume Classification System
CAI 6605 - Trustworthy AI Systems - Midterm Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class Config:
    """Optimized configuration for resume classification"""
    
    # Model Configuration
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 512
    
    # Training Parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 25
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Data Configuration
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Paths
    DATA_PATH = 'data/raw/Resume.csv'
    MODEL_SAVE_PATH = 'models/resume_classifier'
    GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1QWJo26V-95XF1uGJKKVnnf96uaclAENk'
    
    @staticmethod
    def display_config():
        """Display configuration for presentation"""
        print("=" * 70)
        print("📋 PROJECT CONFIGURATION")
        print("=" * 70)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Max Length: {Config.MAX_LENGTH} tokens")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Train/Val/Test Split: 70%/15%/15%")
        print("=" * 70 + "\n")
