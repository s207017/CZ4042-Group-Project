"""
Domain adaptation adapter layers for sentiment analysis.
"""
import torch
import torch.nn as nn
from typing import Optional


class DomainAdapter(nn.Module):
    """
    Domain adaptation layer for transfer learning.
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        """
        Initialize domain adapter.
        
        Args:
            hidden_size: Size of input hidden states
            adapter_size: Size of adapter bottleneck
        """
        super().__init__()
        
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size)
            
        Returns:
            Adapted tensor
        """
        # Down project
        down = self.down_project(x)
        down = self.activation(down)
        
        # Up project
        up = self.up_project(down)
        
        # Residual connection
        return x + up


class AdversarialDiscriminator(nn.Module):
    """
    Adversarial discriminator for domain adaptation.
    """
    
    def __init__(self, hidden_size: int, num_domains: int = 2):
        """
        Initialize discriminator.
        
        Args:
            hidden_size: Size of input features
            num_domains: Number of domains to distinguish
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_domains)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Domain logits
        """
        return self.classifier(x)

