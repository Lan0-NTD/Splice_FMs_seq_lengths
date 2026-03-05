"""
Lightweight 3-class splicing site classifier
Trained on pre-extracted embeddings (from foundation models)

Classes:
- 0: Non-splicing sites (Other)
- 1: Splicing donor site (5'ss)
- 2: Splicing acceptor site (3'ss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpliceSiteClassifier(nn.Module):
    """
    Fully-connected classifier for 3-class splicing site prediction
    Input: Pre-extracted embeddings [batch_size, embedding_dim]
    Output: Class logits [batch_size, 3]
    """
    
    def __init__(self, embedding_dim, num_classes=3, hidden_dims=None, dropout_rate=0.3):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of output classes (default: 3)
            hidden_dims: List of hidden layer dimensions
                        If None: [512, 256]
            dropout_rate: Dropout probability
        """
        super(SpliceSiteClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build network
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Embedding tensor [batch_size, embedding_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.network(x)
    
    def get_predictions(self, logits):
        """Get class predictions and probabilities from logits"""
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        confs = torch.max(probs, dim=1)[0]
        return preds, probs, confs
    
    def __repr__(self):
        return (f"SpliceSiteClassifier("
                f"embedding_dim={self.embedding_dim}, "
                f"num_classes={self.num_classes}, "
                f"dropout={self.dropout_rate})")


def create_classifier(embedding_dim, num_classes=3, hidden_dims=None, dropout_rate=0.3):
    """Factory function to create classifier"""
    return SpliceSiteClassifier(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
