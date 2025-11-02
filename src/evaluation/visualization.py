"""
Visualization utilities for sentiment analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import os


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str] = None,
                         save_path: str = None, title: str = "Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: Names of classes
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or ['Negative', 'Positive'],
                yticklabels=class_names or ['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history.get('train_loss', []), label='Train Loss')
    axes[0].plot(history.get('val_loss', []), label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_f1_vs_dataset_size(results: Dict[str, float], save_path: str = None):
    """
    Plot F1 score vs dataset size (for small dataset experiments).
    
    Args:
        results: Dictionary mapping dataset sizes to F1 scores
        save_path: Path to save the plot
    """
    sizes = sorted(results.keys())
    f1_scores = [results[s] for s in sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, f1_scores, marker='o')
    plt.xlabel('Dataset Size')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Dataset Size')
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

