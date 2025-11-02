"""
Data loading utilities for sentiment analysis datasets.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def load(self):
        """Load dataset. To be implemented by subclasses."""
        raise NotImplementedError


class IMDBDataLoader(DatasetLoader):
    """Loader for IMDB movie reviews dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self, binary: bool = True) -> Tuple[List[str], List[int]]:
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


class YelpDataLoader(DatasetLoader):
    """Loader for Yelp reviews dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self, sample_size: Optional[int] = None, binary: bool = True) -> Tuple[List[str], List[int]]:
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


class SST2Loader(DatasetLoader):
    """Loader for Stanford Sentiment Treebank v2 (SST-2)."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sentences_path = self.data_dir / "datasetSentences.txt"
        self.splits_path = self.data_dir / "datasetSplit.txt"
        self.dictionary_path = self.data_dir / "dictionary.txt"
        self.labels_path = self.data_dir / "sentiment_labels.txt"
        
    def load(self) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """
        Load SST-2 dataset with binary labels.
        
        Returns:
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
        """
        # Load sentences
        sentences_df = pd.read_csv(self.sentences_path, sep="\t")
        sentences_df.columns = ["sentence_index", "sentence"]
        
        # Load splits
        splits_df = pd.read_csv(self.splits_path, sep=",")
        splits_df.columns = ["sentence_index", "split_label"]
        
        # Load dictionary (phrase -> phrase_id)
        dictionary_df = pd.read_csv(self.dictionary_path, sep="|", header=None)
        dictionary_df.columns = ["phrase", "phrase_id"]
        
        # Load sentiment labels (phrase_id -> sentiment_score)
        labels_df = pd.read_csv(self.labels_path, sep="|")
        labels_df.columns = ["phrase_id", "sentiment_score"]
        
        # Merge dictionary with labels
        phrase_labels_df = dictionary_df.merge(labels_df, on="phrase_id")
        
        # Create mapping from phrase to sentiment
        phrase_to_sentiment = dict(zip(
            phrase_labels_df["phrase"],
            phrase_labels_df["sentiment_score"]
        ))
        
        # Merge sentences with splits
        merged_df = sentences_df.merge(splits_df, on="sentence_index")
        
        # Map sentences to sentiment scores
        def get_sentiment(phrase):
            return phrase_to_sentiment.get(phrase, 0.5)
        
        merged_df["sentiment_score"] = merged_df["sentence"].apply(get_sentiment)
        
        # Convert to binary labels (>= 0.5 = positive = 1, < 0.5 = negative = 0)
        merged_df["label"] = (merged_df["sentiment_score"] >= 0.5).astype(int)
        
        # Split by dataset
        train_df = merged_df[merged_df["split_label"] == 1]
        test_df = merged_df[merged_df["split_label"] == 2]
        dev_df = merged_df[merged_df["split_label"] == 3]
        
        return (
            train_df["sentence"].tolist(),
            train_df["label"].tolist(),
            dev_df["sentence"].tolist(),
            dev_df["label"].tolist(),
            test_df["sentence"].tolist(),
            test_df["label"].tolist()
        )


def load_preprocessed_data(dataset_name: str, 
                          data_dir: str = "intermediate/data") -> Tuple[List[str], List[int]]:
    """
    Load preprocessed data from intermediate directory.
    
    Args:
        dataset_name: Name of dataset and split, e.g.:
            - 'imdb_train', 'imdb_val', 'imdb_test'
            - 'yelp_train', 'yelp_val', 'yelp_test'
            - 'sst2_train', 'sst2_val', 'sst2_test'
        data_dir: Directory containing preprocessed files
        
    Returns:
        texts, labels
    """
    data_path = Path(data_dir)
    file_map = {
        # IMDB splits
        'imdb_train': 'imdb_train_preprocessed.csv',
        'imdb_val': 'imdb_val_preprocessed.csv',
        'imdb_test': 'imdb_test_preprocessed.csv',
        # Yelp splits
        'yelp_train': 'yelp_train_preprocessed.csv',
        'yelp_val': 'yelp_val_preprocessed.csv',
        'yelp_test': 'yelp_test_preprocessed.csv',
        # SST-2 splits
        'sst2_train': 'sst2_train_preprocessed.csv',
        'sst2_val': 'sst2_val_preprocessed.csv',
        'sst2_test': 'sst2_test_preprocessed.csv'
    }
    
    if dataset_name not in file_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}.\n"
            f"Choose from: {sorted(file_map.keys())}\n"
            f"Note: All datasets are now split into train/val/test. "
            f"Use '{{dataset}}_train', '{{dataset}}_val', or '{{dataset}}_test'."
        )
    
    file_path = data_path / file_map[dataset_name]
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {file_path}\n"
            f"Please run notebook 02_preprocessing.ipynb first to generate preprocessed data."
        )
    
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    return texts, labels


def create_train_test_split(texts: List[str], labels: List[int], 
                          test_size: float = 0.2, val_size: float = 0.1,
                          random_state: int = 42) -> Tuple[List[str], List[str], List[str], 
                                                           List[int], List[int], List[int]]:
    """
    Create train/val/test split for data.
    
    Args:
        texts: List of text samples
        labels: List of labels
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining after test)
        random_state: Random seed
        
    Returns:
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    """
    # First split: train+val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: train vs val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_size / (1 - test_size),  # Adjust val_size for remaining data
        random_state=random_state,
        stratify=train_val_labels
    )
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

