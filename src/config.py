"""
Centralized configuration for training and experiments.
"""

import os

# ==================== Paths ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "data.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")

MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
BASELINE_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "baseline")
ROBERTA_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "roberta")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ==================== Model ====================
MODEL_NAME = "roberta-base"
NUM_LABELS = 2
LABEL_MAP = {"real": 0, "fake": 1}
LABEL_MAP_INV = {0: "real", 1: "fake"}

# ==================== Training Hyperparameters ====================
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SEED = 42

# ==================== Regularization ====================
DROPOUT_RATE = 0.1
ATTENTION_DROPOUT = 0.1
MAX_GRAD_NORM = 1.0

# ==================== Early Stopping ====================
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_DELTA = 0.001

# ==================== MC Dropout (Uncertainty Estimation) ====================
MC_DROPOUT_RUNS = 10

# ==================== Temperature Scaling (Calibration) ====================
TEMP_SCALING_LR = 0.01
TEMP_SCALING_MAX_ITER = 50

# ==================== Confidence-based Routing ====================
CONFIDENCE_THRESHOLD = 0.7

# ==================== LLM Verification ====================
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1

# ==================== Baseline (LR + TF-IDF) ====================
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)

# ==================== Logging ====================
LOGGING_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 500

# ==================== Device ====================
DEVICE = "cuda"  # Auto-detected in utils.get_device()
