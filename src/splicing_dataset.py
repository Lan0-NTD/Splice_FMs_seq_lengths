"""
Dataset loader for pre-extracted embeddings
Loads embeddings and labels from .pt files
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """
    Load pre-extracted embeddings from .pt file
    
    File format:
        {
            "embeddings": Tensor[N, embedding_dim],
            "labels": Tensor[N]
        }
    """
    
    def __init__(self, embeddings, labels, transform=None):
        """
        Args:
            embeddings: Tensor of shape [N, embedding_dim]
            labels: Tensor of shape [N]
            transform: Optional preprocessing function
        """
        self.embeddings = embeddings
        self.labels = labels
        self.transform = transform
        
        assert len(embeddings) == len(labels), "Mismatch between embeddings and labels"
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        return embedding, label
    
    @staticmethod
    def load_from_file(file_path):
        """Load embeddings and labels from .pt file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
        data = torch.load(file_path)
        
        embeddings = data['embeddings']
        labels = data['labels']
        
        logger.info(f"Loaded embeddings from {file_path}")
        logger.info(f"  Shape: {embeddings.shape}")
        logger.info(f"  Labels: {labels.shape}")
        logger.info(f"  Label distribution: {torch.bincount(labels.long())}")
        
        return EmbeddingDataset(embeddings, labels)


def create_embedding_dataloader(
    embeddings,
    labels,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
):
    """
    Create a DataLoader for embeddings
    
    Args:
        embeddings: Tensor[N, embedding_dim]
        labels: Tensor[N]
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Pin memory for GPU
    
    Returns:
        DataLoader
    """
    dataset = EmbeddingDataset(embeddings, labels)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
