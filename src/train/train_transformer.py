"""
Training script for transformer models.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, List

from ..models.transformer_model import BERTForSentiment, RoBERTaForSentiment
from .trainer_utils import create_dataloader, compute_metrics


def train_transformer(model, train_loader, val_loader, device, 
                     num_epochs: int = 5, learning_rate: float = 2e-5,
                     warmup_steps: float = 0.1, max_grad_norm: float = 1.0):
    """
    Train a transformer model.
    
    Args:
        model: Transformer model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: Torch device
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps (fraction of total)
        max_grad_norm: Gradient clipping norm
        
    Returns:
        Training history
    """
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    num_warmup = int(warmup_steps * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_transformer(model, val_loader, device, criterion)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            # Model saving would go here
    
    return history


def evaluate_transformer(model, dataloader, device, criterion) -> Dict[str, float]:
    """Evaluate transformer model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

