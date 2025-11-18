[file name]: config.py
[file content begin]
"""
Enhanced configuration for Resume Classification System with Bias Detection
CAI 6605 - Trustworthy AI Systems - Final Project
Group 15: Nithin Palyam, Lorenzo LaPlace
"""

class Config:
    """Optimized configuration for resume classification with bias detection"""
    
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
    
    # Bias Analysis Configuration
    DEMOGRAPHIC_FEATURES = ['gender', 'diversity_background', 'privilege_level']
    FAIRNESS_THRESHOLD = 0.8  # Minimum fairness score (0-1)
    BIAS_MITIGATION_STRATEGIES = ['preprocessing', 'inprocessing', 'postprocessing']
    
    # Name substitution experiments
    GENDER_NAMES = {
        'male': ['James', 'Robert', 'John', 'Michael', 'David', 'William', 'Richard'],
        'female': ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan']
    }
    
    ETHNICITY_NAMES = {
        'white': ['Emily', 'Matthew', 'Daniel', 'Emma', 'Olivia'],
        'black': ['Lakisha', 'Jamal', 'Latoya', 'Tyrone', 'Shanice'],
        'hispanic': ['Juan', 'Maria', 'Jose', 'Luisa', 'Carlos'],
        'asian': ['Wei', 'Li', 'Zhang', 'Wang', 'Liu']
    }
    
    @staticmethod
    def display_config():
        """Display configuration for presentation"""
        print("=" * 70)
        print("📋 FINAL PROJECT CONFIGURATION - BIAS-AWARE RESUME CLASSIFICATION")
        print("=" * 70)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Max Length: {Config.MAX_LENGTH} tokens")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Fairness Threshold: {Config.FAIRNESS_THRESHOLD}")
        print(f"Bias Mitigation Strategies: {Config.BIAS_MITIGATION_STRATEGIES}")
        print("=" * 70 + "\n")
[file content end]
