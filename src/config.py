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

# Model configurations (Tối ưu riêng cho Human Genome)
MODELS_CONFIG = {
    "HyenaDNA": {
        "versions": ["tiny-1k", "small-32k"],
        "model_ids": [
            "LongSafari/hyenadna-small-32k-seqlen-hf", # Hỗ trợ cực tốt cho window_size 10000 (deep intronic regions).
            "LongSafari/hyenadna-medium-160k-seqlen-hf", # Hỗ trợ tốt cho window_size 1000 (exonic regions).
        ],
        "source": "huggingface",
        "note": "Native human DNA models. Excellent for understanding long-range human genomic dependencies."
    },
    "DNABert": {
        "versions": ["dnabert2-117m"],
        "model_ids": [
            "zhihan1996/DNABERT-2-117M", 
        ],
        "source": "huggingface",
        "note": "Multi-species but heavily benchmarked on Human GRCh38. Uses BPE tokenization (much better than k-mer for splicing)."
    },
    "NucleotideTransformer": {
        "versions": ["500m-human", "100m-v2"],
        "model_ids": [
            "InstaDeepAI/nucleotide-transformer-500m-human-ref", # CỰC KỲ KHUYÊN DÙNG: Model 500M tham số train thuần trên Human Reference Genome.
            "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", # Model v2 kiến trúc mới, nhẹ và chạy rất nhanh trên RTX 5080.
            
            # Lưu ý: InstaDeep có bản "InstaDeepAI/nucleotide-transformer-2.5b-1000g" 
            # (Train trên 1000 bộ gen người) nhưng nặng 2.5 Tỷ tham số, khả năng cao sẽ Out of Memory với window_size > 1000.
        ],
        "source": "huggingface",
        "note": "Included specific weights trained on the Human Reference Genome."
    }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 0.0001,
    "weight_decay": 0.001,
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

# ============================================================
# SPLICING CLASSIFICATION CONFIGURATION
# ============================================================

EMBEDDING_CONFIG = {
    "method": "center",  # center, mean, or cls
    "max_length": 10000, # TĂNG LÊN để không bị cắt cụt data ở window 10000 (Dynamic padding sẽ tự lo phần dư)
    "batch_size": 128,   # Base batch size 
    "device": "cuda",
    "use_fp16": True,    # Bật lại FP16 vì code mới đã dùng torch.amp an toàn
    "num_workers": 4,    # Giảm từ 16 xuống 4 hoặc 8. 16 sẽ làm nghẽn CPU của bạn.
    
    # CHIẾN LƯỢC BATCH SIZE SỐNG CÒN CHO RTX 5080 (16GB VRAM)
    "batch_size_by_window": {
        300: 256,       # Sequence ngắn -> Batch lớn vô tư
        600: 128,       # Sequence vừa -> Giảm một nửa
        1000: 64,       # Sequence dài -> Giảm tiếp để tránh OOM
        10000: 4,       # ĐẶC BIỆT LƯU Ý: 10000 bp cực kỳ tốn VRAM, chỉ chạy batch 4 hoặc 8.
    }
}

# Embedding storage paths
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Splicing classification
SPLICING_CLASS_NAMES = {
    0: "Other",
    1: "Donor",
    2: "Acceptor"
}

SPLICING_NUM_CLASSES = 3

# Splicing classifier training config
SPLICING_TRAINER_CONFIG = {
    "embedding_dim": None,  # Will be inferred from embeddings
    "num_classes": 3,
    "hidden_dims": [512, 256],
    "dropout_rate": 0.3,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 10,
    "num_folds": 5,
    "cv_random_state": 42,
}

# Results and logging paths for splicing
SPLICING_RESULTS_DIR = RESULTS_DIR / "classifiers"
SPLICING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLICING_TENSORBOARD_DIR = LOGS_DIR / "splicing_classifiers"
SPLICING_TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
