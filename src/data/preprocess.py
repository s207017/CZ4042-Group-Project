"""
Text preprocessing utilities for sentiment analysis.
"""
import re
from typing import List
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    """
    Basic text cleaning: remove HTML tags, extra whitespace, etc.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize_texts(texts: List[str], tokenizer_name: str = 'bert-base-uncased', 
                   max_length: int = 512, return_tensors: str = 'pt'):
    """
    Tokenize texts using a pre-trained tokenizer.
    
    Args:
        texts: List of text strings
        tokenizer_name: Name of the tokenizer model
        max_length: Maximum sequence length
        return_tensors: Format to return ('pt', 'tf', 'np', or None)
        
    Returns:
        Tokenized inputs
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensors
    )


def create_vocabulary(texts: List[str], min_freq: int = 2) -> dict:
    """
    Create vocabulary from texts.
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        Dictionary mapping words to indices
    """
    from collections import Counter
    
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Filter by minimum frequency
    vocab = {word: idx + 2 for idx, (word, count) 
             in enumerate(word_counts.most_common()) if count >= min_freq}
    
    # Add special tokens
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    
    return vocab

