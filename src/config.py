"""
Configuration file for Foundation Model DNA Analysis
"""
import os
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = PROJECT_DIR / "gencode_multi_seq_length"
GTEX_DATA_DIR = PROJECT_DIR / "gtex_multi_seq_length"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = PROJECT_DIR / "plots"
LOGS_DIR = PROJECT_DIR / "logs"

# Create directories if not exist
for dir_path in [PROCESSED_DATA_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
WINDOW_SIZES = [300, 600, 1000, 10000]
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15
TEST_CHROMOSOMES = [20, 21]

# Model configurations
MODELS_CONFIG = {
    "HyenaDNA": {
        "versions": ["tiny", "small", "medium"],
        "model_ids": [
            "gpt2",  # Placeholder - HyenaDNA models not publicly available
            "bert-base-uncased",
            "roberta-base"
        ],
        "source": "huggingface",
        "note": "Original HyenaDNA models require special access. Using alternative models for demo."
    },
    "DNABert": {
        "versions": ["bert-mini", "bert-small", "bert-base"],
        "model_ids": [
            "bert-base-uncased",  # Alternative to zhihan1996/DNA_bert_3
            "bert-base-cased",    # Alternative to zhihan1996/DNA_bert_4
            "roberta-base"        # Alternative to zhihan1996/DNA_bert_5
        ],
        "source": "huggingface",
        "note": "Original DNABert models require special access. Using similar models for demo."
    },
    "NucleotideTransformer": {
        "versions": ["gpt2", "distilbert", "roberta"],
        "model_ids": [
            "gpt2",                  # Placeholder 
            "distilbert-base-uncased",  # Placeholder
            "roberta-base"           # Placeholder
        ],
        "source": "huggingface",
        "note": "Original Nucleotide Transformer models not publicly available. Using similar models for demo."
    }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-5,
    "weight_decay": 1e-6,
    "num_folds": 5,  # for cross-validation
    "device": "cuda",  # or "cpu"
    "seed": 42,
    "early_stopping_patience": 5,
    "num_classes": 3,  # Number of splicing types
}

# Task configuration (classification task for DNA sequences)
TASK_TYPE = "classification"  # or "regression", "sequence_labeling", etc.
NUM_CLASSES = 3  # Splicing types: 0, 1, 2 (multiclass classification)

# Paths for saving models and results
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Data preparation paths
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.pkl"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.pkl"
GENCODE_TEST_DATA_PATH = PROCESSED_DATA_DIR / "gencode_test_data.pkl"
GTEX_TEST_DATA_PATH = PROCESSED_DATA_DIR / "gtex_test_data.pkl"

# Results paths
RESULTS_SUMMARY_PATH = RESULTS_DIR / "results_summary.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.json"

# Logging configuration
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
