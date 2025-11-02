"""
Baseline models for sentiment analysis.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class LSTMModel(nn.Module):
    """LSTM-based sentiment classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, 
                 hidden_dim: int = 128, num_layers: int = 2, 
                 num_classes: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Class logits
        """
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # For bidirectional
        output = self.fc(self.dropout(hidden))
        
        return output


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


class BiLSTMWithAttention(nn.Module):
    """BiLSTM with attention mechanism for sentiment classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global max pooling
        pooled = torch.max(attn_out, dim=1)[0]
        output = self.fc(self.dropout(pooled))
        
        return output

