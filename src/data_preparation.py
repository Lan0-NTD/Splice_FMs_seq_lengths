"""
Data preparation module for DNA sequences
Handles loading, splitting, and saving training/validation/test datasets
"""

import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DNADataPreparation:
    """Prepare DNA sequence data for model training"""
    
    def __init__(self, 
                 raw_data_dir: Path,
                 gtex_data_dir: Path,
                 processed_data_dir: Path,
                 window_sizes: List[int] = [300, 600, 1000, 2000, 10000]):
        """
        Initialize data preparation
        
        Args:
            raw_data_dir: Path to gencode data directory
            gtex_data_dir: Path to gtex data directory
            processed_data_dir: Path to save processed data
            window_sizes: List of window sizes to process
        """
        self.raw_data_dir = raw_data_dir
        self.gtex_data_dir = gtex_data_dir
        self.processed_data_dir = processed_data_dir
        self.window_sizes = window_sizes
        
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_gencode_data(self, window_size: int) -> pd.DataFrame:
        """Load gencode data for a specific window size"""
        file_path = self.raw_data_dir / f"gencode{window_size}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading gencode data: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} samples from gencode{window_size}")
        return df
    
    def load_gtex_data(self, window_size: int) -> pd.DataFrame:
        """Load gtex data for a specific window size"""
        file_path = self.gtex_data_dir / f"gtex{window_size}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading gtex data: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded {len(df)} samples from gtex{window_size}")
        return df
    
    def split_by_chromosome(self,
                           df: pd.DataFrame,
                           test_chromosomes: List[int] = [20, 21],
                           train_split: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by chromosome for train/val/test
        
        Args:
            df: Input dataframe with 'CHROM' column
            test_chromosomes: Chromosomes to use for test set
            train_split: Ratio for train vs val split (e.g., 0.85 means 85% train, 15% val)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.copy()  # Avoid modify original df
        
        # Convert CHROM to numeric if needed
        if df['CHROM'].dtype == 'object':
            # Remove 'chr' prefix
            df['CHROM'] = df['CHROM'].str.replace('chr', '', regex=False)
            
            # Map sex chromosomes and mitochondrial DNA to numeric values
            # Standard mapping: X=23, Y=24, M/MT=25
            chromosome_map = {
                'X': 23,
                'Y': 24,
                'M': 25,
                'MT': 25
            }
            
            # Replace mapped chromosomes
            df['CHROM'] = df['CHROM'].replace(chromosome_map)
            
            # Convert remaining to numeric
            df['CHROM'] = pd.to_numeric(df['CHROM'], errors='coerce')
            
            # Remove any remaining NaN values if conversion failed
            initial_len = len(df)
            df = df.dropna(subset=['CHROM'])
            if len(df) < initial_len:
                logger.info(f"Filtered out {initial_len - len(df)} rows with invalid chromosome values")
            
            # Convert to int
            df['CHROM'] = df['CHROM'].astype(int)
        
        # Separate test set
        test_mask = df['CHROM'].isin(test_chromosomes)
        test_df = df[test_mask].copy()
        
        # Separate train+val set
        train_val_df = df[~test_mask].copy()
        
        # Split train+val into train and val
        stratify_labels = train_val_df['Splicing_types'] if 'Splicing_types' in train_val_df.columns else None
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=(1 - train_split),
            random_state=42,
            stratify=stratify_labels
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_gencode_data(self,
                            window_sizes: List[int] = None,
                            test_chromosomes: List[int] = [20, 21],
                            train_split: float = 0.85,
                            save: bool = True) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Prepare gencode data (split into train/val/test)
        
        Args:
            window_sizes: Window sizes to process
            test_chromosomes: Chromosomes for test set
            train_split: Train/val split ratio
            save: Whether to save processed data
        
        Returns:
            Dictionary with split data for each window size
        """
        if window_sizes is None:
            window_sizes = self.window_sizes
        
        all_data = {}
        
        for window_size in window_sizes:
            try:
                df = self.load_gencode_data(window_size)
                train_df, val_df, test_df = self.split_by_chromosome(
                    df,
                    test_chromosomes=test_chromosomes,
                    train_split=train_split
                )
                
                all_data[window_size] = {
                    'train': train_df,
                    'val': val_df,
                    'test': test_df,
                    'window_size': window_size,
                    'data_source': 'gencode'
                }
                
            except Exception as e:
                logger.error(f"Error processing window size {window_size}: {str(e)}")
        
        if save:
            self.save_processed_data(all_data, 'gencode')
        
        return all_data
    
    def prepare_gtex_data(self,
                         window_sizes: List[int] = None,
                         save: bool = True) -> Dict[int, pd.DataFrame]:
        """
        Prepare gtex data (used as test set)
        
        Args:
            window_sizes: Window sizes to process
            save: Whether to save processed data
        
        Returns:
            Dictionary with gtex data for each window size
        """
        if window_sizes is None:
            window_sizes = self.window_sizes
        
        all_data = {}
        
        for window_size in window_sizes:
            try:
                df = self.load_gtex_data(window_size)
                all_data[window_size] = {
                    'data': df,
                    'window_size': window_size,
                    'data_source': 'gtex',
                    'split': 'test'
                }
                
            except Exception as e:
                logger.error(f"Error processing gtex window size {window_size}: {str(e)}")
        
        if save:
            self.save_processed_data(all_data, 'gtex')
        
        return all_data
    
    def save_processed_data(self, data: Dict, data_source: str):
        """
        Save processed data to pickle files
        
        Args:
            data: Dictionary with processed data
            data_source: Either 'gencode' or 'gtex'
        """
        output_file = self.processed_data_dir / f"{data_source}_processed.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved processed {data_source} data to {output_file}")
    
    def load_processed_data(self, data_source: str) -> Dict:
        """
        Load previously saved processed data
        
        Args:
            data_source: Either 'gencode' or 'gtex'
        
        Returns:
            Dictionary with processed data
        """
        input_file = self.processed_data_dir / f"{data_source}_processed.pkl"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_file}")
        
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded processed {data_source} data from {input_file}")
        return data
    
    def get_data_statistics(self, df: pd.DataFrame, data_name: str = "") -> Dict[str, Any]:
        """
        Get statistics about a dataset
        
        Args:
            df: Input dataframe
            data_name: Name of dataset for logging
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_samples': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
        }
        
        # Get chromosome distribution if available
        if 'CHROM' in df.columns:
            stats['chromosome_distribution'] = df['CHROM'].value_counts().to_dict()
        
        logger.info(f"Statistics for {data_name}:\n{stats}")
        return stats
    
    def prepare_all_data(self, save: bool = True):
        """
        Prepare all data (gencode and gtex) with statistics
        
        Args:
            save: Whether to save processed data
        
        Returns:
            Dictionary with all prepared data and statistics
        """
        logger.info("="*50)
        logger.info("Starting data preparation")
        logger.info("="*50)
        
        # Prepare gencode data
        gencode_data = self.prepare_gencode_data(save=save)
        
        # Prepare gtex data
        gtex_data = self.prepare_gtex_data(save=save)
        
        # Collect statistics
        stats = {}
        
        for window_size in self.window_sizes:
            if window_size in gencode_data:
                stats[f'gencode_{window_size}'] = {
                    'train': self.get_data_statistics(gencode_data[window_size]['train'], 
                                                     f'gencode_{window_size}_train'),
                    'val': self.get_data_statistics(gencode_data[window_size]['val'],
                                                   f'gencode_{window_size}_val'),
                    'test': self.get_data_statistics(gencode_data[window_size]['test'],
                                                    f'gencode_{window_size}_test'),
                }
            
            if window_size in gtex_data:
                stats[f'gtex_{window_size}'] = self.get_data_statistics(gtex_data[window_size]['data'],
                                                                        f'gtex_{window_size}')
        
        logger.info("="*50)
        logger.info("Data preparation completed")
        logger.info("="*50)
        
        return {
            'gencode': gencode_data,
            'gtex': gtex_data,
            'statistics': stats
        }
