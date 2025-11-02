"""Evaluation and metrics module."""
from .metrics import compute_metrics, calculate_confusion_matrix
from .error_analysis import analyze_errors
from .visualization import plot_confusion_matrix, plot_training_history

__all__ = [
    'compute_metrics',
    'calculate_confusion_matrix',
    'analyze_errors',
    'plot_confusion_matrix',
    'plot_training_history'
]

