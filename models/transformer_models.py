"""
Transformer-based models for sentiment analysis.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional


class BERTForSentiment(nn.Module):
    """BERT-based model for sentiment classification."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2, 
                 dropout: float = 0.1, freeze_bert: bool = False):
        """
        Initialize BERT model for sentiment analysis.
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT weights
        """
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (for BERT)
            
        Returns:
            Class logits
        """
        outputs = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return self.classifier(cls_output)


class RoBERTaForSentiment(nn.Module):
    """RoBERTa-based model for sentiment classification."""
    
    def __init__(self, model_name: str = 'roberta-base', num_classes: int = 2, 
                 dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        
        self.roberta = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


class XLNetForSentiment(nn.Module):
    """XLNet-based model for sentiment classification."""
    
    def __init__(self, model_name: str = 'xlnet-base-cased', num_classes: int = 2,
                 dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        
        self.xlnet = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.xlnet.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.d_model // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.xlnet(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, -1, :]
        return self.classifier(cls_output)


class ELECTRAForSentiment(nn.Module):
    """ELECTRA-based model for sentiment classification."""
    
    def __init__(self, model_name: str = 'google/electra-base-discriminator', 
                 num_classes: int = 2, dropout: float = 0.1, freeze_base: bool = False):
        super().__init__()
        
        self.electra = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.electra.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

