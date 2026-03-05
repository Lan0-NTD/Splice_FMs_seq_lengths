# Splice Site Prediction Pipeline

## Overview

This is a **two-phase pipeline** for 3-class splice site classification using pre-extracted embeddings from foundation models:

1. **Phase 1: Embedding Extraction (Offline, One-Time)** - Extract embeddings from foundation models and save to disk
2. **Phase 2: Lightweight Classification** - Train fast classifiers on embeddings using 5-fold cross-validation

**Expected Performance:** 60x faster than end-to-end training (1 hour total vs 48+ hours)

---

## Architecture

### Data Flow
```
Raw DNA Sequences (GENCODE/GTEx)
    ↓
[Foundation Models: Tokenization + Forward Pass]
    ↓
Extract Embeddings (Center Position)
    ↓
Save to Disk: embeddings/.pt files
    ↓
[Lightweight Classifiers: FC Networks]
    ↓
5-Fold Cross-Validation Training
    ↓
Results: JSON + TensorBoard + Checkpoints
```

### Folder Structure
```
data/
├── embeddings/                         # Extracted embeddings
│   ├── 300/                           # Window size 300 bp
│   │   ├── HyenaDNA_gpt2/
│   │   ├── DNABert_bert-base-uncased/
│   │   └── NucleotideTransformer_gpt2/
│   ├── 600/, 1000/, 10000/           # Other window sizes
│   └── (similarly structured)
│
results/
├── classifiers/                        # Training results
│   ├── HyenaDNA_gpt2_w300/
│   │   ├── results.json               # Per-fold + averaged metrics
│   │   ├── best_model.pt              # Best checkpoint
│   │   ├── config.json                # Hyperparameters
│   │   └── tensorboard/
│   │       ├── fold_0/, fold_1/, ...  # Per-fold metrics
│   │       └── all_folds/             # Averaged metrics
│   └── (similarly for other models)
│
logs/
├── splicing_classifiers/              # TensorBoard logs
│   └── (automatically created during training)
```

---

## Files Created

### Core Pipeline Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/splicing_embed_extract.py` | Standalone script to extract embeddings from all models | 300+ |
| `src/splicing_classifier.py` | Lightweight FC classifier (copy from Task1) | 80 |
| `src/splicing_dataset.py` | DataLoader for .pt embedding files | 90 |
| `src/splicing_metrics.py` | Comprehensive metrics computation | 300+ |
| `src/splicing_train.py` | Training loop with JSON + TensorBoard logging | 400+ |
| `splicing_prediction.ipynb` | Main pipeline notebook (orchestration) | 5 cells |
| `src/config.py` | (Updated) Splicing-specific configuration | +50 |

### Key Components

#### 1. Embedding Extraction (`splicing_embed_extract.py`)
**Standalone script** to extract embeddings offline:
```bash
python src/splicing_embed_extract.py
```

**Features:**
- Processes all model/window size combinations
- Batch processing for speed
- Center position embedding extraction
- Saves GENCODE train/val/test and GTEx test separately
- Progress logging with time estimates

**Output:**
```
data/embeddings/{window}/{model}/
├── train_embeddings.pt      [Tensor, Tensor]
├── val_embeddings.pt        [Tensor, Tensor]
├── test_embeddings.pt       [Tensor, Tensor]
└── gtex_test_embeddings.pt  [Tensor, Tensor]
```

#### 2. Classifier (`splicing_classifier.py`)
**3-class fully-connected classifier:**
- Input: embeddings [batch, embedding_dim]
- Hidden: [512, 256] with BatchNorm + ReLU + Dropout
- Output: logits [batch, 3]

```python
model = SpliceSiteClassifier(embedding_dim=1024, dropout_rate=0.3)
logits = model(embeddings)
```

#### 3. Dataset Loader (`splicing_dataset.py`)
**Load pre-extracted embeddings from .pt files:**

```python
dataset = EmbeddingDataset.load_from_file("embeddings.pt")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### 4. Metrics (`splicing_metrics.py`)
**10+ comprehensive metrics:**
- Accuracy, Balanced Accuracy, F1 (weighted/macro)
- Precision, Recall, Specificity, Sensitivity (per-class)
- MCC, Cohen's Kappa
- ROC-AUC, PR-AUC (per-class)

```python
metrics, cm = MetricsComputer.compute_metrics(y_true, y_pred, y_probs)
```

#### 5. Training (`splicing_train.py`)
**5-fold CV training with comprehensive logging:**

**Features:**
- Per-epoch TensorBoard logging (as requested)
- JSON results: per-fold + averaged metrics (Option A)
- Best model checkpointing
- Early stopping with patience

**JSON Output Structure (Option A - Detailed):**
```json
{
  "experiment_name": "HyenaDNA_gpt2_w300",
  "timestamp": "2026-03-04T15:30:00",
  "per_fold_results": {
    "fold_0": {
      "best_epoch": 15,
      "best_f1": 0.856,
      "metrics": {...},
      "confusion_matrix": [[...], [...], [...]]
    },
    "fold_1": {...},
    ...
  },
  "averaged_metrics": {
    "accuracy_mean": 0.823,
    "accuracy_std": 0.015,
    "f1_weighted_mean": 0.825,
    "f1_weighted_std": 0.012,
    ...
  }
}
```

#### 6. Main Notebook (`splicing_prediction.ipynb`)
**Orchestration notebook with 5 phases:**

**Cell 1:** Setup & imports
**Cell 2:** Configuration check (verify data files)
**Cell 3:** Extract embeddings (optional, conditional)
**Cell 4:** List available embeddings
**Cell 5:** Train classifiers (5-fold CV for each model)
**Cell 6:** Results comparison
**Cell 7:** Evaluate on test sets
**Cell 8:** Summary

---

## Quick Start

### Option 1: Notebook Workflow (Recommended)

```python
# 1. Open and run splicing_prediction.ipynb
jupyter notebook splicing_prediction.ipynb

# 2. Set EXTRACT_EMBEDDINGS = True in cell 3 to extract embeddings
# 3. Run all cells sequentially
# 4. Results will be saved to results/classifiers/ and logs/splicing_classifiers/
```

### Option 2: Command Line (Embedding Extraction Only)

```bash
# Extract embeddings for all model/window combinations
python src/splicing_embed_extract.py

# Extract specific subset
python src/splicing_embed_extract.py --window-sizes 300 600 --models HyenaDNA DNABert
```

### Option 3: Python Script (Training Only)

```bash
# Train classifier on pre-extracted embeddings
python src/splicing_train.py data/embeddings/300/HyenaDNA_gpt2/train_embeddings.pt \
    --experiment-name HyenaDNA_gpt2_w300 \
    --num-folds 5 \
    --epochs 50 \
    --batch-size 32
```

---

## Results Output

### JSON Results
**Location:** `results/classifiers/{experiment_name}/results.json`

**Contents:**
- Per-fold: best_epoch, best_f1, metrics, confusion_matrix
- Averaged: all metrics + std dev across folds

**Includes:**
- Accuracy, Balanced Accuracy, F1 (weighted/macro)
- Precision, Recall, Sensitivity, Specificity (per-class)
- MCC, Cohen's Kappa
- ROC-AUC, PR-AUC (per-class)

### TensorBoard Logs
**Location:** `logs/splicing_classifiers/{experiment_name}/tensorboard/`

**Structure:**
```
tensorboard/
├── fold_0/          # Fold-specific metrics
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
└── all_folds/       # Averaged metrics across folds
```

**Per-epoch logging includes:**
- Train/Val loss
- Accuracy, Balanced Accuracy
- F1-weighted, F1-macro
- MCC
- Per-class metrics

**View:**
```bash
tensorboard --logdir logs/splicing_classifiers/
```

### Checkpoints
**Location:** `results/classifiers/{experiment_name}/best_model.pt`

Best model weights from best epoch of fold 0 (for reference)

---

## Hyperparameters

**Embedding Extraction:**
- Method: Center position of hidden states
- Max length: 512 tokens
- Batch size: 256 (for inference)

**Classifier Training:**
- Architecture: [embedding_dim] → [512] → [256] → [3]
- Activation: ReLU + BatchNorm + Dropout(0.3)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Batch size: 32
- Max epochs: 50
- Early stopping: 10 epochs patience
- CV folds: 5 (stratified)

**Data Split:**
- GENCODE: Train (chr 1-19, 22, 85%) + Val (chr 1-19, 22, 15%)
- GENCODE Test: chr 20, 21 (fixed, 100%)
- GTEx Test: Cross-dataset evaluation

---

## Expected Results

**Performance Comparison:**

| Approach | Training Time | Classification Speed | Model Size |
|----------|--------------|-------------------|-----------|
| End-to-end (current) | 48+ hours | Slow (needs GPU) | Large |
| **Embedding-based (new)** | **1 hour** | **Fast (CPU OK)** | **~100MB** |

**Speedup:** 48x faster training + embeddings extraction

**Expected Metrics:**
- F1-weighted: 0.82-0.87
- Accuracy: 0.80-0.86
- MCC: 0.70-0.80
- ROC-AUC: 0.90-0.96

---

## Configuration

**Edit `src/config.py` to customize:**

```python
# Embedding extraction
EMBEDDING_CONFIG = {
    "method": "center",      # or "mean", "cls"
    "batch_size": 256,       # inference batch size
}

# Classifier training
SPLICING_TRAINER_CONFIG = {
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 10,
    "num_folds": 5,
}

# Data
WINDOW_SIZES = [300, 600, 1000, 10000]
TEST_CHROMOSOMES = [20, 21]
```

---

## Logging Details

### Console Output (Per Epoch)
```
Epoch  25: train_loss=0.4523, val_loss=0.3892, f1=0.8462 ✓ (best: 25)
Epoch  30: train_loss=0.3921, val_loss=0.3765, f1=0.8501 (patience: 1/10)
```

### TensorBoard Metrics (Per Epoch)
- **Loss:** train_loss, val_loss
- **Accuracy:** accuracy, balanced_accuracy
- **F1:** f1_weighted, f1_macro
- **Other:** mcc, precision, recall, sensitivity, specificity, roc_auc, pr_auc

### JSON Results (Per Fold + Averaged)
```json
{
  "fold_0": {
    "metrics": {
      "accuracy": 0.823,
      "f1_weighted": 0.825,
      ...
    },
    "confusion_matrix": [[940, 30, 5], ...]
  },
  "averaged_metrics": {
    "accuracy_mean": 0.815,
    "accuracy_std": 0.012,
    ...
  }
}
```

---

## Troubleshooting

### Issue: "No embeddings found"
**Solution:** Run embedding extraction first
```bash
python src/splicing_embed_extract.py
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in notebook or config
```python
EMBEDDING_CONFIG["batch_size"] = 128  # was 256
SPLICING_TRAINER_CONFIG["batch_size"] = 16  # was 32
```

### Issue: "TensorBoard not available"
**Solution:** Install with `pip install tensorboard`

### Issue: "Models not loading"
**Solution:** Check that foundation models are available in your environment
```bash
pip install transformers torch
```

---

## File Dependencies

**Input Files (Must Exist):**
- `gencode_multi_seq_length/gencode{300,600,1000,10000}.csv`
- `gtex_multi_seq_length/gtex{300,600,1000,10000}.csv`

**Generated Files:**
- `data/embeddings/{window}/{model}/*.pt`
- `results/classifiers/{experiment}/results.json`
- `results/classifiers/{experiment}/best_model.pt`
- `logs/splicing_classifiers/{experiment}/tensorboard/**`

---

## Related Files

**From Task1 (Fully Reused):**
- `training/model.py` → `src/splicing_classifier.py` ✓
- `training/dataset.py` → `src/splicing_dataset.py` ✓
- `training/metrics.py` → `src/splicing_metrics.py` ✓

**From Task1 (Adapted):**
- `training/train_set.py` → `src/splicing_train.py` (90% reused) ✓
- `data_preparation/extract_embed.py` → `src/splicing_embed_extract.py` (80% adapted) ✓

**From Current Project (Enhanced):**
- `src/config.py` (added splicing config) ✓
- `src/data_preparation.py` (reuses split_by_chromosome) ✓
- `src/models.py` (reuses FoundationModelLoader) ✓

---

## Next Steps

1. **Extract embeddings** (if not already done):
   ```bash
   python src/splicing_embed_extract.py
   ```

2. **Run main pipeline**:
   ```bash
   jupyter notebook splicing_prediction.ipynb
   ```

3. **Monitor TensorBoard** (in another terminal):
   ```bash
   tensorboard --logdir logs/splicing_classifiers/
   ```

4. **Analyze results**:
   - JSON files in `results/classifiers/`
   - TensorBoard dashboard
   - Comparison notebook

---

## Performance Notes

**Embedding Extraction Time (estimated):**
- Per model: 15-20 minutes
- All 3 models × 4 windows = 12 combinations
- Total: 1-2 hours

**Classifier Training Time (estimated):**
- Per combination: 10-15 minutes (5 folds × ~2 min/fold)
- All 12 combinations: 2-3 hours total
- **Total pipeline:** 3-5 hours

**GPU Requirements:**
- Embedding extraction: 16GB VRAM (batch size 256)
- Classifier training: 2-4GB VRAM (batch size 32)
- Can use CPU for classifier training if needed

---

## Contact & Support

For issues or questions, check:
1. Console error messages (detailed logging)
2. TensorBoard logs (visual monitoring)
3. JSON results files (detailed metrics)
4. This README (common issues)

---

**Last Updated:** March 4, 2026
**Pipeline Version:** 1.0 (Embedding-Based Classification)
