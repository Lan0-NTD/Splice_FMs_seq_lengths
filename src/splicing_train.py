"""
Training script for splice site classification using pre-extracted embeddings

Features:
- 5-fold stratified cross-validation
- Per-epoch logging
- JSON results storage (per-fold + averaged)
- TensorBoard metrics visualization
- Best model checkpointing
- Confusion matrix logging
"""

import sys
import json
import copy
import random
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    logging.warning("TensorBoard not available, metrics will only be saved to JSON")
    SummaryWriter = None

# Add src to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "src"))

from splicing_classifier import SpliceSiteClassifier
from splicing_dataset import EmbeddingDataset
from splicing_metrics import MetricsComputer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpliceClassifierTrainer:
    """Trainer for splice site classification"""
    
    def __init__(self, embedding_dim, num_classes=3, device='cuda', results_dir='results/classifiers'):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of output classes
            device: 'cuda' or 'cpu'
            results_dir: Directory to save results
        """
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_model = None
        self.best_epoch = None
        self.best_score = -np.inf
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch_idx=None, num_epochs=None, fold_idx=None, num_folds=None):
        """Train one epoch"""
        model.train()
        total_loss = 0.0

        epoch_label = "Epoch"
        if epoch_idx is not None and num_epochs is not None:
            epoch_label = f"Epoch {epoch_idx}/{num_epochs}"
        if fold_idx is not None and num_folds is not None:
            epoch_label = f"Fold {fold_idx}/{num_folds} | {epoch_label}"

        batch_iterator = tqdm(
            train_loader,
            desc=epoch_label,
            leave=False,
            dynamic_ncols=True
        )

        for embeddings, labels in batch_iterator:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device).long()
            
            # Forward pass
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_iterator.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @staticmethod
    def set_random_seed(seed: int = 42, deterministic: bool = False, seed_cuda: bool = True):
        """Set random seeds for near-stable reproducibility.

        When CUDA is in a bad state, use CPU-only seeding to avoid triggering
        implicit CUDA calls inside torch.manual_seed.
        """
        random.seed(seed)
        np.random.seed(seed)

        if seed_cuda and torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.random.default_generator.manual_seed(seed)

        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    def _ensure_device_ready(self):
        """Probe CUDA once and fallback to CPU if runtime state is unhealthy."""
        if self.device.type != 'cuda':
            return

        try:
            _ = torch.cuda.current_device()
            probe = torch.tensor([1.0], device=self.device)
            probe = probe * 2.0
            _ = float(probe.item())
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(
                "CUDA runtime is not healthy (%s). Falling back to CPU for this training run.",
                e,
            )
            self.device = torch.device('cpu')
    
    def eval_epoch(self, model, val_loader, criterion):
        """Evaluate one epoch"""
        model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).long()
                
                logits = model(embeddings)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Compute metrics
        metrics, cm = MetricsComputer.compute_metrics(all_labels, all_preds, all_probs)
        
        return avg_loss, metrics, cm, all_labels, all_preds, all_probs
    
    def train_with_cv(self, embeddings, labels, experiment_name, num_folds=5,
                      num_epochs=50, batch_size=32, learning_rate=1e-3, 
                      weight_decay=1e-5, early_stopping_patience=10,
                      seed=42, deterministic=False):
        """
        Train classifier with k-fold cross-validation
        
        Args:
            embeddings: [N, embedding_dim]
            labels: [N]
            experiment_name: Name for this experiment (used in output paths)
            num_folds: Number of cross-validation folds
            num_epochs: Maximum epochs per fold
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Epochs to wait before early stopping
            seed: Random seed for near-stable reproducibility
            deterministic: If True, stricter determinism (slower)
        
        Returns:
            Dictionary with results per fold and averaged results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING: {experiment_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        logger.info(f"Num folds: {num_folds}, Epochs: {num_epochs}, Batch size: {batch_size}")

        self._ensure_device_ready()
        self.set_random_seed(
            seed=seed,
            deterministic=deterministic,
            seed_cuda=(self.device.type == 'cuda')
        )
        
        # Setup experiment directory
        exp_dir = self.results_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = exp_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        tensorboard_dir = exp_dir / 'tensorboard'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # K-fold split
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        
        # Results storage
        fold_results = {}
        all_fold_metrics = []
        
        # Track global best fold for easy inference selection
        global_best_fold_idx = None
        global_best_fold_mcc = -np.inf
        global_best_checkpoint_name = None

        # Loop through folds
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
            logger.info(f"\n{'#'*80}")
            logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
            logger.info(f"{'#'*80}")
            
            # Split data
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
            logger.info(f"Train label dist: {np.bincount(y_train)}")
            logger.info(f"Val label dist: {np.bincount(y_val)}")
            
            # Create datasets and loaders
            train_dataset = EmbeddingDataset(X_train, y_train)
            val_dataset = EmbeddingDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(self.device.type == 'cuda')
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=(self.device.type == 'cuda')
            )
            
            # Create model
            model = SpliceSiteClassifier(
                embedding_dim=self.embedding_dim,
                num_classes=self.num_classes,
                hidden_dims=[512, 256],
                dropout_rate=0.3
            ).to(self.device)
            
            # Optimizer and criterion
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            # TensorBoard writer
            tb_writer = None
            if SummaryWriter is not None:
                fold_tb_dir = tensorboard_dir / f'fold_{fold_idx}'
                tb_writer = SummaryWriter(str(fold_tb_dir))
            
            # Training loop
            best_epoch = -1
            best_val_mcc = -np.inf
            patience_counter = 0
            best_model_state = None
            fold_history = {
                'train_loss': [],
                'val_loss': [],
                'metrics': []
            }
            
            for epoch in range(num_epochs):
                # Train
                train_loss = self.train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    epoch_idx=epoch + 1,
                    num_epochs=num_epochs,
                    fold_idx=fold_idx + 1,
                    num_folds=num_folds,
                )
                
                # Validate
                val_loss, metrics, cm, _, _, _ = self.eval_epoch(model, val_loader, criterion)
                
                # Metrics used for logging and model selection
                val_f1 = metrics['f1_weighted']
                val_acc = metrics.get('accuracy', 0.0)
                val_mcc = metrics.get('mcc', 0.0)
                pr_auc_values = [
                    metrics.get(f'pr_auc_{class_name}', np.nan)
                    for class_name in MetricsComputer.CLASS_NAMES
                ]
                val_auprc = float(np.nanmean(pr_auc_values)) if len(pr_auc_values) > 0 else 0.0
                if np.isnan(val_auprc):
                    val_auprc = 0.0
                
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['metrics'].append(metrics)
                
                # Log to TensorBoard (PER EPOCH as requested)
                if tb_writer is not None:
                    tb_writer.add_scalar('Loss/train', train_loss, epoch)
                    tb_writer.add_scalar('Loss/val', val_loss, epoch)
                    tb_writer.add_scalar('Metrics/f1_weighted', val_f1, epoch)
                    tb_writer.add_scalar('Metrics/accuracy', metrics['accuracy'], epoch)
                    tb_writer.add_scalar('Metrics/balanced_accuracy', metrics['balanced_accuracy'], epoch)
                    tb_writer.add_scalar('Metrics/mcc', metrics['mcc'], epoch)
                    tb_writer.add_scalar('Metrics/auprc_macro', val_auprc, epoch)
                
                # Early stopping
                if val_mcc > best_val_mcc:
                    best_val_mcc = val_mcc
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    best_model_state = copy.deepcopy(model.state_dict())
                    epoch_status = f"✓ (best by mcc: {best_epoch+1})"
                else:
                    patience_counter += 1
                    epoch_status = f"(patience: {patience_counter}/{early_stopping_patience})"

                logger.info(
                    f"Epoch {epoch+1:3d}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"acc={val_acc:.4f}, f1={val_f1:.4f}, mcc={val_mcc:.4f}, auprc={val_auprc:.4f} "
                    f"{epoch_status}"
                )

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience exceeded)")
                    break
                
                lr_scheduler.step()
            
            # Close TensorBoard writer
            if tb_writer is not None:
                tb_writer.close()
            
            # Final evaluation on validation set
            logger.info(f"\nFold {fold_idx + 1} - Evaluating best model (epoch {best_epoch+1})...")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                self.best_model = best_model_state
            val_loss, metrics, cm, val_labels, val_preds, val_probs = self.eval_epoch(
                model, val_loader, criterion
            )
            
            # Save fold results
            fold_number = fold_idx + 1
            checkpoint_name = f"best_fold{fold_number}.pt"
            checkpoint_path = checkpoints_dir / checkpoint_name

            if best_model_state is not None:
                checkpoint_payload = {
                    'fold_idx': fold_idx,
                    'fold_number': fold_number,
                    'best_epoch': best_epoch + 1,
                    'best_metric': 'mcc',
                    'best_mcc': float(best_val_mcc),
                    'embedding_dim': int(self.embedding_dim),
                    'num_classes': int(self.num_classes),
                    'experiment_name': experiment_name,
                    'model_state_dict': best_model_state,
                    'hyperparameters': {
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'early_stopping_patience': early_stopping_patience,
                        'seed': seed,
                        'deterministic': deterministic,
                    },
                }
                torch.save(checkpoint_payload, checkpoint_path)

            if best_val_mcc > global_best_fold_mcc:
                global_best_fold_mcc = float(best_val_mcc)
                global_best_fold_idx = fold_idx
                global_best_checkpoint_name = checkpoint_name

            fold_results[f'fold_{fold_idx}'] = {
                'best_epoch': best_epoch + 1,
                'best_metric': 'mcc',
                'best_mcc': float(best_val_mcc),
                'best_f1_at_best_epoch': float(fold_history['metrics'][best_epoch]['f1_weighted']) if best_epoch >= 0 else 0.0,
                'best_checkpoint': checkpoint_name,
                'val_loss': float(val_loss),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                           for k, v in metrics.items()},
                'confusion_matrix': cm.tolist()
            }
            
            all_fold_metrics.append(metrics)
            
            logger.info(f"✓ Fold {fold_idx + 1} completed")
            logger.info(f"  Best epoch: {best_epoch + 1}")
            logger.info(f"  Best MCC: {best_val_mcc:.4f}")
            logger.info(f"  Validation accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Validation balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"  Validation MCC: {metrics['mcc']:.4f}")
            fold_pr_auc_values = [
                metrics.get(f'pr_auc_{class_name}', np.nan)
                for class_name in MetricsComputer.CLASS_NAMES
            ]
            fold_auprc = float(np.nanmean(fold_pr_auc_values)) if len(fold_pr_auc_values) > 0 else 0.0
            if np.isnan(fold_auprc):
                fold_auprc = 0.0
            logger.info(f"  Validation AUPRC (macro): {fold_auprc:.4f}")
        
        # Compute averaged metrics across folds
        logger.info(f"\n{'='*80}")
        logger.info(f"CROSS-VALIDATION SUMMARY")
        logger.info(f"{'='*80}")
        
        averaged_metrics = {}
        metric_names = MetricsComputer.get_metric_names()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_fold_metrics if metric_name in m]
            if values:
                averaged_metrics[f'{metric_name}_mean'] = float(np.mean(values))
                averaged_metrics[f'{metric_name}_std'] = float(np.std(values))
        
        # Log averaged metrics to TensorBoard
        if SummaryWriter is not None:
            all_folds_tb_dir = tensorboard_dir / 'all_folds'
            all_folds_writer = SummaryWriter(str(all_folds_tb_dir))
            for metric_name, value in averaged_metrics.items():
                all_folds_writer.add_scalar(f'Metrics/{metric_name}', value, 0)
            all_folds_writer.close()
        
        # Save comprehensive results to JSON (Option A: Detailed per-fold + average)
        results_summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'embedding_dim': int(self.embedding_dim),
            'num_folds': num_folds,
            'best_fold': {
                'fold_idx': int(global_best_fold_idx) if global_best_fold_idx is not None else None,
                'fold_number': int(global_best_fold_idx + 1) if global_best_fold_idx is not None else None,
                'best_metric': 'mcc',
                'best_mcc': float(global_best_fold_mcc) if global_best_fold_idx is not None else None,
                'checkpoint': global_best_checkpoint_name,
                'checkpoint_path': str(checkpoints_dir / global_best_checkpoint_name) if global_best_checkpoint_name is not None else None,
            },
            'hyperparameters': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'early_stopping_patience': early_stopping_patience,
                'seed': seed,
                'deterministic': deterministic
            },
            'per_fold_results': fold_results,
            'averaged_metrics': averaged_metrics
        }
        
        results_file = exp_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"✓ Results saved to: {results_file}")
        
        # Print summary
        logger.info(f"\nAveraged Metrics Across All Folds:")
        logger.info(f"{'='*80}")
        for key, value in sorted(averaged_metrics.items()):
            if isinstance(value, (int, float)):
                logger.info(f"  {key:40s} {value:8.4f}")
        logger.info(f"{'='*80}")
        
        return results_summary


def main():
    """Main entry point for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train splice site classifier")
    parser.add_argument('embeddings_file', type=str, help='Path to embeddings .pt file')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--results-dir', type=str, default='results/classifiers', help='Results directory')
    
    args = parser.parse_args()
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings_file}")
    data = torch.load(args.embeddings_file)
    embeddings = data['embeddings']
    labels = data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']
    
    embedding_dim = embeddings.shape[1]
    
    # Experiment name
    experiment_name = args.experiment_name or Path(args.embeddings_file).stem
    
    # Train
    trainer = SpliceClassifierTrainer(
        embedding_dim=embedding_dim,
        device=args.device,
        results_dir=args.results_dir
    )
    
    results = trainer.train_with_cv(
        embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings,
        labels,
        experiment_name=experiment_name,
        num_folds=args.num_folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n✓ Training completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
