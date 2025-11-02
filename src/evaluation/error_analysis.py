"""
Error analysis utilities.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def analyze_errors(texts: List[str], predictions: np.ndarray, 
                   labels: np.ndarray, model_name: str = "model") -> pd.DataFrame:
    """
    Analyze prediction errors.
    
    Args:
        texts: List of text samples
        predictions: Predicted labels
        labels: True labels
        model_name: Name of the model
        
    Returns:
        DataFrame with error analysis
    """
    errors = []
    
    for text, pred, label in zip(texts, predictions, labels):
        if pred != label:
            errors.append({
                'text': text,
                'predicted': pred,
                'actual': label,
                'error_type': 'FP' if pred == 1 and label == 0 else 'FN',
                'text_length': len(text.split())
            })
    
    error_df = pd.DataFrame(errors)
    
    if len(error_df) > 0:
        print(f"\nError Analysis for {model_name}:")
        print(f"Total errors: {len(error_df)}")
        print(f"False Positives: {len(error_df[error_df['error_type'] == 'FP'])}")
        print(f"False Negatives: {len(error_df[error_df['error_type'] == 'FN'])}")
        print(f"Average text length: {error_df['text_length'].mean():.2f}")
    
    return error_df


def find_common_errors(error_df: pd.DataFrame, top_n: int = 10) -> Dict:
    """
    Find common patterns in errors.
    
    Args:
        error_df: DataFrame from analyze_errors
        top_n: Number of patterns to return
        
    Returns:
        Dictionary of common error patterns
    """
    # Placeholder for more sophisticated analysis
    return {
        'common_words': [],
        'common_phrases': [],
        'length_distribution': error_df['text_length'].describe().to_dict()
    }

