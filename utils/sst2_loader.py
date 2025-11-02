"""
Proper SST-2 data loader with correct binary labels.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


class SST2Loader:
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

