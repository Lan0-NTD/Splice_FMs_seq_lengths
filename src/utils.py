"""
Utility functions for DNA sequence analysis
"""

import json
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manage and aggregate results from multiple models"""
    
    def __init__(self, results_dir: Path, plots_dir: Path):
        """
        Initialize results manager
        
        Args:
            results_dir: Directory with results
            plots_dir: Directory to save plots
        """
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_results(self, model_name: str, model_id: str) -> Dict[str, Any]:
        """Load results for a specific model"""
        results_path = self.results_dir / f"{model_name}_{model_id}" / "results.json"
        
        if not results_path.exists():
            logger.warning(f"Results not found for {model_name}_{model_id}")
            return {}
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results
    
    def aggregate_cv_results(self, models_config: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Aggregate cross-validation results from all models
        
        Args:
            models_config: Configuration with model information
        
        Returns:
            Aggregated results dictionary
        """
        aggregated = {}
        
        for model_name, config in models_config.items():
            aggregated[model_name] = {}
            
            for model_id in config['model_ids']:
                results = self.load_model_results(model_name, model_id)
                
                if results and 'fold_results' in results:
                    fold_results = results['fold_results']
                    
                    # Calculate averages
                    accuracies = []
                    f1_scores = []
                    
                    for fold in fold_results:
                        if 'best_metrics' in fold:
                            metrics = fold['best_metrics']
                            accuracies.append(metrics.get('accuracy', 0))
                            f1_scores.append(metrics.get('f1', 0))
                    
                    aggregated[model_name][model_id] = {
                        'mean_accuracy': np.mean(accuracies) if accuracies else 0,
                        'std_accuracy': np.std(accuracies) if accuracies else 0,
                        'mean_f1': np.mean(f1_scores) if f1_scores else 0,
                        'std_f1': np.std(f1_scores) if f1_scores else 0,
                        'num_folds': len(fold_results)
                    }
        
        return aggregated
    
    def save_aggregated_results(self, aggregated_results: Dict[str, Any]):
        """Save aggregated results to file"""
        output_path = self.results_dir / "aggregated_results.json"
        
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
        
        serializable_results = convert_to_serializable(aggregated_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"Saved aggregated results to {output_path}")
    
    def plot_model_comparison(self, aggregated_results: Dict[str, Any], metric: str = 'accuracy'):
        """
        Plot comparison of models
        
        Args:
            aggregated_results: Aggregated results from all models
            metric: Metric to plot ('accuracy' or 'f1')
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        models = []
        values = []
        errors = []
        colors = []
        color_map = {'HyenaDNA': 'skyblue', 'DNABert': 'orange', 'NucleotideTransformer': 'green'}
        
        for model_name, model_results in aggregated_results.items():
            for model_id, metrics in model_results.items():
                models.append(f"{model_name}\n{model_id}")
                
                if metric == 'accuracy':
                    values.append(metrics['mean_accuracy'])
                    errors.append(metrics['std_accuracy'])
                else:  # f1
                    values.append(metrics['mean_f1'])
                    errors.append(metrics['std_f1'])
                
                colors.append(color_map.get(model_name, 'gray'))
        
        x_pos = np.arange(len(models))
        ax.bar(x_pos, values, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Mean {metric.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / f"model_comparison_{metric}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {plot_path}")
        plt.close()
    
    def plot_performance_by_window_size(self, results_by_window: Dict[int, Dict]) -> None:
        """
        Plot performance across different window sizes
        
        Args:
            results_by_window: Dictionary with results for each window size
        """
        window_sizes = sorted(results_by_window.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for model_info, metrics in results_by_window.items():
            window = model_info[0]
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            
            if window in window_sizes:
                axes[0].plot(window, accuracy, 'o-', label=str(window))
                axes[1].plot(window, f1, 's-', label=str(window))
        
        axes[0].set_xlabel('Window Size', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy vs Window Size', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        axes[1].set_xlabel('Window Size', fontsize=12)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('F1 Score vs Window Size', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / "performance_by_window_size.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved window size plot to {plot_path}")
        plt.close()
    
    def create_results_summary_table(self, aggregated_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary table of results
        
        Args:
            aggregated_results: Aggregated results from all models
        
        Returns:
            DataFrame with results summary
        """
        data = []
        
        for model_name, model_results in aggregated_results.items():
            for model_id, metrics in model_results.items():
                data.append({
                    'Model Family': model_name,
                    'Model ID': model_id,
                    'Mean Accuracy': f"{metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}",
                    'Mean F1 Score': f"{metrics['mean_f1']:.4f} ± {metrics['std_f1']:.4f}",
                    'Num Folds': metrics['num_folds']
                })
        
        df = pd.DataFrame(data)
        return df
    
    def export_results_summary(self, aggregated_results: Dict[str, Any]) -> None:
        """Export results summary to CSV and JSON"""
        # Create summary table
        df = self.create_results_summary_table(aggregated_results)
        
        # Save to CSV
        csv_path = self.results_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results summary to {csv_path}")
        
        # Save to JSON
        json_path = self.results_dir / "results_summary.json"
        self.save_aggregated_results(aggregated_results)
        logger.info(f"Saved results summary to {json_path}")


class DataPreparationTracker:
    """Track data preparation process and save state"""
    
    def __init__(self, output_dir: Path):
        """Initialize tracker"""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_data_state(self, data_state: Dict[str, Any], name: str = "data_state"):
        """
        Save data preparation state
        
        Args:
            data_state: Dictionary with data information
            name: Name for the state file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.output_dir / f"{name}_{timestamp}.json"
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return str(obj)
        
        serializable_state = convert_to_serializable(data_state)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_state, f, indent=4)
        
        logger.info(f"Saved data state to {file_path}")
    
    def load_data_state(self, file_path: Path) -> Dict[str, Any]:
        """Load previously saved data state"""
        with open(file_path, 'r') as f:
            data_state = json.load(f)
        
        logger.info(f"Loaded data state from {file_path}")
        return data_state


def setup_logging(log_dir: Path, log_file: str = "training.log"):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Logging initialized")
