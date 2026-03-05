"""
Comprehensive metrics for 3-class splicing site classification

Metrics computed:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1 (weighted & per-class)
- MCC (Matthews Correlation Coefficient)
- Specificity, Sensitivity per class
- ROC-AUC (One-vs-Rest)
- PR-AUC (One-vs-Rest)
- Top-k accuracy
- Cohen's Kappa
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, cohen_kappa_score, top_k_accuracy_score
)
import logging

from config import SPLICING_CLASS_NAMES

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Compute comprehensive metrics for classification"""
    
    NUM_CLASSES = len(SPLICING_CLASS_NAMES)
    CLASS_NAMES = [SPLICING_CLASS_NAMES[idx] for idx in range(NUM_CLASSES)]
    CLASS_LABELS = list(range(NUM_CLASSES))
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_probs=None, average='weighted'):
        """
        Compute all metrics
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
            y_probs: Predicted probabilities [N, num_classes] (optional)
            average: 'weighted', 'macro', 'micro'
        
        Returns:
            Dictionary of metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Per-class metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class precision/recall/f1
        precisions = precision_score(y_true, y_pred, average=None, labels=MetricsComputer.CLASS_LABELS, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, labels=MetricsComputer.CLASS_LABELS, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, labels=MetricsComputer.CLASS_LABELS, zero_division=0)
        
        for i, class_name in enumerate(MetricsComputer.CLASS_NAMES):
            metrics[f'precision_{class_name}'] = precisions[i]
            metrics[f'recall_{class_name}'] = recalls[i]
            metrics[f'f1_{class_name}'] = f1s[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=MetricsComputer.CLASS_LABELS)
        
        # Specificity and Sensitivity (TPR) per class
        for i in range(MetricsComputer.NUM_CLASSES):
            # True Positive Rate (Sensitivity/Recall)
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'sensitivity_{MetricsComputer.CLASS_NAMES[i]}'] = sensitivity
            
            # True Negative Rate (Specificity)
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'specificity_{MetricsComputer.CLASS_NAMES[i]}'] = specificity
        
        # ROC-AUC and PR-AUC (One-vs-Rest)
        if y_probs is not None and len(np.unique(y_true)) > 1:
            y_probs = np.asarray(y_probs)
            
            try:
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true, y_probs, average='macro', multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_probs, average='weighted', multi_class='ovr'
                )
                
                # Per-class AUC
                for i in range(MetricsComputer.NUM_CLASSES):
                    y_true_binary = (y_true == i).astype(int)
                    try:
                        auc_score = roc_auc_score(y_true_binary, y_probs[:, i])
                        metrics[f'roc_auc_{MetricsComputer.CLASS_NAMES[i]}'] = auc_score
                    except:
                        metrics[f'roc_auc_{MetricsComputer.CLASS_NAMES[i]}'] = 0.0
            except:
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_weighted'] = 0.0
                for i, name in enumerate(MetricsComputer.CLASS_NAMES):
                    metrics[f'roc_auc_{name}'] = 0.0
            
            # PR-AUC
            try:
                for i in range(MetricsComputer.NUM_CLASSES):
                    y_true_binary = (y_true == i).astype(int)
                    precision_vals, recall_vals, _ = precision_recall_curve(
                        y_true_binary, y_probs[:, i]
                    )
                    pr_auc = auc(recall_vals, precision_vals)
                    metrics[f'pr_auc_{MetricsComputer.CLASS_NAMES[i]}'] = pr_auc
            except:
                for i, name in enumerate(MetricsComputer.CLASS_NAMES):
                    metrics[f'pr_auc_{name}'] = 0.0
        
        # Top-k accuracy
        if y_probs is not None:
            try:
                metrics['top1_accuracy'] = top_k_accuracy_score(y_true, y_probs, k=1)
                if MetricsComputer.NUM_CLASSES >= 2:
                    metrics['top2_accuracy'] = top_k_accuracy_score(y_true, y_probs, k=2)
            except:
                metrics['top1_accuracy'] = metrics['accuracy']
                metrics['top2_accuracy'] = metrics['accuracy']
        
        return metrics, cm
    
    @staticmethod
    def get_metric_names(include_confusion_matrix=False):
        """Get list of all metric names"""
        metric_names = [
            'accuracy', 'balanced_accuracy', 'cohen_kappa', 'mcc',
            'precision_macro', 'precision_weighted', 'f1_macro', 'f1_weighted',
            'recall_macro', 'recall_weighted',
        ]
        
        for class_name in MetricsComputer.CLASS_NAMES:
            metric_names.extend([
                f'precision_{class_name}',
                f'recall_{class_name}',
                f'sensitivity_{class_name}',
                f'specificity_{class_name}',
                f'f1_{class_name}',
                f'roc_auc_{class_name}',
                f'pr_auc_{class_name}'
            ])
        
        metric_names.extend(['roc_auc_macro', 'roc_auc_weighted', 'top1_accuracy', 'top2_accuracy'])
        
        if include_confusion_matrix:
            metric_names.append('confusion_matrix')
        
        return metric_names
    
    @staticmethod
    def format_metrics(metrics, decimal_places=4):
        """Format metrics for display"""
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted[key] = round(value, decimal_places)
            else:
                formatted[key] = value
        return formatted
    
    @staticmethod
    def print_metrics(metrics, title="Metrics"):
        """Pretty print metrics"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        # Group metrics by category
        categories = {
            'Overall': ['accuracy', 'balanced_accuracy', 'mcc', 'cohen_kappa'],
            'Precision': [k for k in metrics.keys() if 'precision' in k],
            'Recall': [k for k in metrics.keys() if 'recall' in k],
            'F1 Score': [k for k in metrics.keys() if 'f1' in k],
            'Sensitivity': [k for k in metrics.keys() if 'sensitivity' in k],
            'Specificity': [k for k in metrics.keys() if 'specificity' in k],
            'ROC-AUC': [k for k in metrics.keys() if 'roc_auc' in k],
            'PR-AUC': [k for k in metrics.keys() if 'pr_auc' in k],
            'Top-K': [k for k in metrics.keys() if 'top' in k],
        }
        
        for category, keys in categories.items():
            keys = [k for k in keys if k in metrics]
            if keys:
                print(f"\n{category}:")
                for key in keys:
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        print(f"  {key:30s} {value:8.4f}")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def compare_metrics(metrics_list, labels=None, top_k='accuracy'):
        """Compare metrics across multiple runs/models"""
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(metrics_list))]
        
        print(f"\n{'='*80}")
        print(f"Metrics Comparison (sorted by {top_k})")
        print(f"{'='*80}\n")
        
        # Get common metric keys
        common_keys = set(metrics_list[0].keys()) & set(
            *[set(m.keys()) for m in metrics_list[1:]]
        )
        common_keys = sorted([k for k in common_keys if isinstance(metrics_list[0][k], (int, float))])
        
        # Sort by top_k metric
        sorted_indices = sorted(
            range(len(metrics_list)),
            key=lambda i: metrics_list[i].get(top_k, 0),
            reverse=True
        )
        
        # Print header
        header = f"{'Model/Fold':<20}"
        for key in common_keys:
            header += f" {key:>12}"
        print(header)
        print("-" * len(header))
        
        # Print rows
        for idx in sorted_indices:
            line = f"{labels[idx]:<20}"
            for key in common_keys:
                value = metrics_list[idx].get(key, 0)
                line += f" {value:12.4f}"
            print(line)
        
        print(f"{'='*80}\n")
