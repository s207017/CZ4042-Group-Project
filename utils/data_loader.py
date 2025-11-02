"""
Data loading utilities for sentiment analysis datasets.
"""
import pandas as pd
import json
import os
from pathlib import Path
from typing import Tuple, List


class SST2DataLoader:
    """Loader for Stanford Sentiment Treebank v2 dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.base_dir = self.data_dir / "SST2-Data" / "SST2-Data" / "stanfordSentimentTreebank" / "stanfordSentimentTreebank"
        
    def load(self) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Load SST-2 dataset from files.
        
        Returns:
            train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels
        """
        # Load sentences
        sentences_df = pd.read_csv(self.base_dir / "datasetSentences.txt", sep="\t")
        
        # Load splits
        splits_df = pd.read_csv(self.base_dir / "datasetSplit.txt", sep=",")
        
        # Load sentiment labels
        labels_df = pd.read_csv(self.base_dir / "sentiment_labels.txt", sep="|")
        labels_df.columns = ["phrase_id", "label"]
        
        # Merge to get sentences with splits
        merged = sentences_df.merge(splits_df, on="sentence_index")
        
        # Load phrases dictionary
        phrases_df = pd.read_csv(self.base_dir / "dictionary.txt", sep="|", header=None)
        phrases_df.columns = ["phrase", "phrase_id"]
        
        # Get phrase IDs for each sentence (needed to match labels)
        # This is a simplified version - SST-2 binary classification
        # uses specific phrase mappings
        
        # For now, return simplified binary labels based on file structure
        # Need to implement proper mapping based on sentence-to-phrase relationships
        
        train_data = merged[merged["splitset_label"] == 1]
        dev_data = merged[merged["splitset_label"] == 3]
        test_data = merged[merged["splitset_label"] == 2]
        
        return (
            train_data["sentence"].tolist(),
            [],
            dev_data["sentence"].tolist(),
            [],
            test_data["sentence"].tolist(),
            []
        )


class IMDBDataLoader:
    """Loader for IMDB movie reviews dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self, binary=True) -> Tuple[List[str], List[int]]:
        """
        Load IMDB dataset from CSV.
        
        Args:
            binary: If True, convert to binary classification (0, 1)
            
        Returns:
            texts, labels
        """
        df = pd.read_csv(self.data_path)
        
        texts = df["review"].tolist()
        # Convert sentiment labels
        labels = (df["sentiment"] == "positive").astype(int).tolist() if binary else df["sentiment"].tolist()
        
        return texts, labels


class YelpDataLoader:
    """Loader for Yelp reviews dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self, sample_size=None, binary=True) -> Tuple[List[str], List[int]]:
        """
        Load Yelp dataset from JSON.
        
        Args:
            sample_size: Number of samples to load (None for all)
            binary: If True, convert to binary classification (3+ stars = positive)
            
        Returns:
            texts, labels
        """
        texts = []
        labels = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                    
                data = json.loads(line)
                texts.append(data["text"])
                
                if binary:
                    labels.append(1 if data["stars"] >= 3 else 0)
                else:
                    labels.append(data["stars"] - 1)  # Map 1-5 stars to 0-4
                    
        return texts, labels


def create_train_test_split(texts: List[str], labels: List[int], 
                          test_size: float = 0.2, random_state: int = 42):
    """
    Create train-test split for data.
    
    Args:
        texts: List of text samples
        labels: List of labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )

