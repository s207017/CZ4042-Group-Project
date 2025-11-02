"""
Data augmentation utilities for sentiment analysis.
Useful for small dataset scenarios.
"""
import random
from typing import List, Tuple


def augment_text(text: str, method: str = "synonym_replacement") -> str:
    """
    Augment a single text sample.
    
    Args:
        text: Input text
        method: Augmentation method
        
    Returns:
        Augmented text
    """
    if method == "synonym_replacement":
        # Placeholder - would use nltk or similar
        return text
    elif method == "back_translation":
        # Placeholder - would use translation APIs
        return text
    elif method == "random_insertion":
        words = text.split()
        if len(words) > 0:
            idx = random.randint(0, len(words))
            words.insert(idx, words[random.randint(0, len(words) - 1)])
        return " ".join(words)
    elif method == "random_deletion":
        words = text.split()
        if len(words) > 1:
            words.pop(random.randint(0, len(words) - 1))
        return " ".join(words)
    else:
        return text


def augment_dataset(texts: List[str], labels: List[int], 
                    augmentation_factor: int = 1) -> Tuple[List[str], List[int]]:
    """
    Augment entire dataset.
    
    Args:
        texts: List of texts
        labels: List of labels
        augmentation_factor: How many augmented samples per original
        
    Returns:
        Augmented texts and labels
    """
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        for _ in range(augmentation_factor):
            aug_text = augment_text(text, method="random_insertion")
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    
    return augmented_texts, augmented_labels

