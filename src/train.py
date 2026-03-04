"""
Training module for foundation models on DNA sequences
Includes cross-validation, evaluation, and TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    logging.warning("TensorBoard not installed")

logger = logging.getLogger(__name__)


class DNASequenceDataset(Dataset):
    """PyTorch Dataset for DNA sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, tokenizer=None, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            sequences: Array of DNA sequences (strings)
            labels: Array of labels
            tokenizer: Tokenizer for encoding sequences
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has pad_token set
        if self.tokenizer is not None:
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # Ensure pad_token is set (double check)
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            encoding = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # If no tokenizer, return encoded sequence
            return {
                'sequence': sequence,
                'label': torch.tensor(label, dtype=torch.long)
            }


class FoundationModelTrainer:
    """Trainer for foundation models"""
    
    def __init__(self,
                 model_name: str,
                 model_id: str,
                 model,
                 tokenizer,
                 config: Dict[str, Any],
                 results_dir: Path,
                 logs_dir: Path,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer
        
        Args:
            model_name: Name of model family (e.g., "HyenaDNA")
            model_id: Model identifier
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            config: Training configuration
            results_dir: Directory to save results
            logs_dir: Directory for TensorBoard logs
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.model_id = model_id
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        self.device = device
        self.num_classes = config.get('num_classes', 3)
        
        # Add classification head to model
        hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
        self.classification_head = nn.Linear(hidden_size, self.num_classes)
        self.classification_head.to(device)
        
        # Create a wrapper model that includes the classification head
        class ModelWithHead(nn.Module):
            def __init__(self, base_model, head):
                super().__init__()
                self.base_model = base_model
                self.head = head
            
            def forward(self, input_ids, attention_mask=None):
                outputs = self.base_model(input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state.mean(dim=1)
                logits = self.head(pooled)
                return logits
        
        self.model = ModelWithHead(self.base_model, self.classification_head)
        self.model.to(device)
        
        # Create model-specific result directory
        self.model_result_dir = results_dir / f"{model_name}_{model_id}"
        self.model_result_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_log_dir = logs_dir / f"{model_name}_{model_id}"
        self.model_log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = None
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'fold_results': []
        }
    
    def create_dataloader(self, sequences: np.ndarray, labels: np.ndarray, 
                         batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader for training"""
        dataset = DNASequenceDataset(sequences, labels, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_epoch(self, train_loader: DataLoader, optimizer, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            if isinstance(batch, dict):
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass - now returns logits directly
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                else:
                    labels = batch['label'].to(self.device)
                    logits = self.model(batch)
                    loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        logits = self.model(input_ids, attention_mask=attention_mask)
                    else:
                        labels = batch['label'].to(self.device)
                        logits = self.model(batch)
                
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        
        return avg_loss, metrics
    
    def train_with_cv(self,
                      sequences: np.ndarray,
                      labels: np.ndarray,
                      window_size: int,
                      data_source: str = "gencode",
                      num_folds: int = 5):
        """
        Train with cross-validation
        
        Args:
            sequences: Array of DNA sequences
            labels: Array of labels
            window_size: Window size of sequences
            data_source: Source of data (gencode or gtex)
            num_folds: Number of CV folds
        """
        logger.info(f"Starting {num_folds}-fold cross-validation")
        logger.info(f"Model: {self.model_name}_{self.model_id}")
        logger.info(f"Data: {data_source}, Window size: {window_size}")
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        all_fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold + 1}/{num_folds}")
            logger.info(f"{'='*50}")
            
            # Initialize TensorBoard writer for this fold
            fold_log_dir = self.model_log_dir / f"fold_{fold}"
            self.writer = SummaryWriter(log_dir=str(fold_log_dir))
            
            # Split data
            train_seqs = sequences[train_idx]
            train_labels = labels[train_idx]
            val_seqs = sequences[val_idx]
            val_labels = labels[val_idx]
            
            # Create dataloaders
            train_loader = self.create_dataloader(train_seqs, train_labels, 
                                                 self.config['batch_size'], shuffle=True)
            val_loader = self.create_dataloader(val_seqs, val_labels,
                                               self.config['batch_size'], shuffle=False)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            # Training loop
            fold_results = {
                'fold': fold,
                'train_losses': [],
                'val_losses': [],
                'best_metrics': {}
            }
            
            patience = 0
            best_val_loss = float('inf')
            
            for epoch in range(self.config['epochs']):
                # Train
                train_loss = self.train_epoch(train_loader, optimizer, epoch)
                fold_results['train_losses'].append(train_loss)
                
                # Validate
                val_loss, metrics = self.validate(val_loader)
                fold_results['val_losses'].append(val_loss)
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Metrics/accuracy', metrics['accuracy'], epoch)
                self.writer.add_scalar('Metrics/f1', metrics['f1'], epoch)
                
                logger.info(f"Epoch {epoch + 1}/{self.config['epochs']} | "
                           f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                           f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    fold_results['best_metrics'] = metrics
                    patience = 0
                else:
                    patience += 1
                
                if patience >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            self.writer.close()
            all_fold_results.append(fold_results)
        
        self.training_history['fold_results'] = all_fold_results
        self._log_cv_summary(all_fold_results)
        
        return all_fold_results
    
    def _log_cv_summary(self, fold_results: List[Dict]):
        """Log cross-validation summary"""
        logger.info(f"\n{'='*50}")
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info(f"{'='*50}")
        
        accuracies = [fr['best_metrics'].get('accuracy', 0) for fr in fold_results]
        f1_scores = [fr['best_metrics'].get('f1', 0) for fr in fold_results]
        
        logger.info(f"Mean Accuracy: {np.mean(accuracies):.4f} "
                   f"(+/- {np.std(accuracies):.4f})")
        logger.info(f"Mean F1 Score: {np.mean(f1_scores):.4f} "
                   f"(+/- {np.std(f1_scores):.4f})")
    
    def save_checkpoint(self, checkpoint_dir: Path = None):
        """Save model checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = self.model_result_dir
        
        checkpoint_path = checkpoint_dir / f"{self.model_name}_{self.model_id}_checkpoint.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'model_id': self.model_id,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_results(self):
        """Save training results"""
        results_path = self.model_result_dir / "results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results = convert_to_serializable(self.training_history)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Saved results to {results_path}")
    
    def save_training_history(self):
        """Save training history to pickle"""
        history_path = self.model_result_dir / "training_history.pkl"
        
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        logger.info(f"Saved training history to {history_path}")
