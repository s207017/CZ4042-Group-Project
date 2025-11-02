"""
Evaluation metrics for sentiment analysis.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple


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
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }


def calculate_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(labels, predictions)


def get_classification_report(predictions: np.ndarray, labels: np.ndarray,
                            target_names: List[str] = None) -> str:
    """
    Get classification report.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        target_names: Names of classes
        
    Returns:
        Classification report string
    """
    return classification_report(labels, predictions, target_names=target_names)

