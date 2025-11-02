"""
Training script for baseline models (LSTM, CNN).
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict

from .trainer_utils import compute_metrics
import numpy as np


def train_baseline(model, train_loader, val_loader, device,
                  num_epochs: int = 10, learning_rate: float = 0.001):
    """
    Train a baseline model (LSTM or CNN).
    
    Args:
        model: Baseline model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: Torch device
        num_epochs: Number of epochs
        learning_rate: Learning rate
        
    Returns:
        Training history
    """
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_baseline(model, val_loader, device, criterion)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    return history


def evaluate_baseline(model, dataloader, device, criterion) -> Dict[str, float]:
    """Evaluate baseline model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

