"""
Training utilities and helper functions.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloader(texts: List[str], labels: List[int], 
                     tokenizer_name: str, batch_size: int = 16,
                     max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.
    
    Args:
        texts: List of text strings
        labels: List of labels
        tokenizer_name: HuggingFace tokenizer name
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = SentimentDataset(texts, labels, tokenizer, max_length)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }

