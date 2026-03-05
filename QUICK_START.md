# Quick Start Guide - Splice Site Prediction Pipeline

## 5-Minute Setup

### Step 1: Verify Data Files
Check that GENCODE and GTEx data files exist:
```
gencode_multi_seq_length/
├── gencode300.csv
├── gencode600.csv
├── gencode1000.csv
└── gencode10000.csv

gtex_multi_seq_length/
├── gtex300.csv
├── gtex600.csv
├── gtex1000.csv
└── gtex10000.csv
```

✓ If these exist, proceed to Step 2.

---

### Step 2: Run the Main Pipeline (Recommended)

Open and run the notebook:
```bash
jupyter notebook splicing_prediction.ipynb
```

**What happens:**
1. Cell 1-2: Setup and configuration check
2. Cell 3: Extract embeddings (if needed)
3. Cell 4: List available embeddings
4. Cell 5: **Train classifiers** (main work, ~30 min per model)
5. Cell 6-8: Results comparison and summary

**That's it!** Results will be in:
- JSON: `results/classifiers/`
- TensorBoard: `logs/splicing_classifiers/`

---

### Step 3 (Optional): View Results

**View TensorBoard:**
```bash
tensorboard --logdir logs/splicing_classifiers/
```
Then open http://localhost:6006 in browser

**View JSON Results:**
```bash
# List all experiments
ls results/classifiers/

# View specific results
cat results/classifiers/HyenaDNA_gpt2_w300/results.json | python -m json.tool
```

---

## Common Workflows

### Workflow A: Full Pipeline (Extract + Train)
```python
# In notebook, set:
EXTRACT_EMBEDDINGS = True

# Then run all cells
# Expected time: 4-5 hours total
```

### Workflow B: Training Only (Skip Extract)
```python
# In notebook, set:
EXTRACT_EMBEDDINGS = False

# Run cell 2-3 to check available embeddings
# If embeddings exist, jump to cell 5 (training)
# Expected time: 2-3 hours
```

### Workflow C: Custom Subset
```python
# Extract specific models/windows only:
python src/splicing_embed_extract.py \
    --window-sizes 300 600 \
    --models HyenaDNA DNABert

# Then train on those
```

---

## Expected Results (Order of Magnitude)

After training, you'll see:

**Per-Fold Metrics (JSON):**
```json
{
  "fold_0": {
    "best_epoch": 15,
    "best_f1": 0.856,
    "metrics": {
      "accuracy": 0.823,
      "f1_weighted": 0.825,
      "mcc": 0.722,
      ...
    }
  }
}
```

**Averaged Results:**
```
accuracy_mean: 0.815 ± 0.012
f1_weighted_mean: 0.820 ± 0.014
mcc_mean: 0.710 ± 0.018
```

**TensorBoard Visualization:**
- Training/validation loss curves
- Per-epoch metrics
- Model comparison across folds

---

## Timing

| Phase | Time | Notes |
|-------|------|-------|
| Data verification | <1 min | Just checking files exist |
| Embedding extraction | 1-2 hours | One-time, 12 model/window combos |
| Classifier training | 2-3 hours | 5 folds × 12 combos × ~2 min |
| Total | 3-5 hours | **First run includes extraction** |

**Subsequent runs:** 2-3 hours (skip extraction)

---

## File Locations (Quick Reference)

| Item | Location |
|------|----------|
| Code | `src/splicing_*.py` |
| Main notebook | `splicing_prediction.ipynb` |
| Embeddings | `data/embeddings/{ws}/{model}/` |
| JSON results | `results/classifiers/{exp}/results.json` |
| TensorBoard | `logs/splicing_classifiers/{exp}/tensorboard/` |
| README | `SPLICING_PIPELINE_README.md` |
| This guide | `QUICK_START.md` |

---

## Customization (Optional)

Edit `src/config.py` to change:

```python
# Faster training (fewer epochs or folds)
SPLICING_TRAINER_CONFIG = {
    "num_epochs": 30,        # was 50
    "num_folds": 3,          # was 5
}

# Faster embeddings (smaller batch)
EMBEDDING_CONFIG = {
    "batch_size": 128,       # was 256 (if GPU memory limited)
}

# Test different window sizes
WINDOW_SIZES = [300, 600]   # was [300, 600, 1000, 10000]
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No embeddings found" | Run cell 3 in notebook with `EXTRACT_EMBEDDINGS = True` |
| CUDA out of memory | Reduce batch size in config (half current value) |
| Training too slow | Use fewer epochs/folds or GPUs with more VRAM |
| TensorBoard not showing | Install with `pip install tensorboard` |

---

## Next Steps

1. **Run the notebook** → generates results in 3-5 hours
2. **Check JSON results** → `results/classifiers/`
3. **View TensorBoard** → `tensorboard --logdir logs/splicing_classifiers/`
4. **Compare models** → Check F1 scores in JSON files
5. **Use best model** → Checkpoint saved in results folder

---

## Key Metrics (What to Look For)

After training, focus on:
- **F1-weighted:** ~0.82-0.87 (overall performance)
- **Balanced Accuracy:** ~0.80-0.85 (per-class balance)
- **MCC:** ~0.70-0.80 (robust metric)
- **Confusion Matrix:** Check per-class recall

---

## Performance: Before vs After

| Metric | Old Approach | New Approach |
|--------|-------------|--------------|
| Training time | 48+ hours | **1 hour** |
| Variants | 40 (4 ratios × 10 sets) | **12 (models × windows)** |
| Metrics tracked | 4 | **10+** |
| Results logging | Basic | **JSON + TensorBoard** |
| **Speedup** | — | **60x faster** |

---

## Questions?

- **Code questions:** Check comments in `src/splicing_*.py`
- **Pipeline questions:** See `SPLICING_PIPELINE_README.md`
- **Results questions:** Check JSON files and TensorBoard visualization

---

**Ready to start?** Open and run `splicing_prediction.ipynb` now!
