"""
CNN-based models for sentiment analysis.
"""
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """CNN-based sentiment classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 num_filters: int = 100, filter_sizes: list = [3, 4, 5],
                 num_classes: int = 2, dropout: float = 0.5):
        """
        Initialize CNN model with multiple filter sizes.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Class logits
        """
        embedded = self.embedding(x).unsqueeze(1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded)).squeeze(3)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.fc(self.dropout(concatenated))
        
        return output

