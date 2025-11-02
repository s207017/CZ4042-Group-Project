"""
Script to preprocess and save all datasets.
"""
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import IMDBDataLoader, create_train_test_split
from utils.sst2_loader import SST2Loader
from utils.preprocessing import clean_text


def preprocess_imdb():
    """Preprocess IMDB dataset."""
    print("Processing IMDB dataset...")
    
    loader = IMDBDataLoader('IMDB Dataset.csv')
    texts, labels = loader.load(binary=True)
    
    # Clean texts
    print("Cleaning texts...")
    texts = [clean_text(text) for text in tqdm(texts)]
    
    # Split into train/val/test
    train_texts, test_texts, train_labels, test_labels = create_train_test_split(
        texts, labels, test_size=0.2
    )
    train_texts, val_texts, train_labels, val_labels = create_train_test_split(
        train_texts, train_labels, test_size=0.1
    )
    
    # Save
    os.makedirs('data/imdb', exist_ok=True)
    
    # Train
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    train_df.to_csv('data/imdb/train.csv', index=False)
    
    # Val
    val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
    val_df.to_csv('data/imdb/val.csv', index=False)
    
    # Test
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    test_df.to_csv('data/imdb/test.csv', index=False)
    
    print(f"IMDB: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")


def preprocess_sst2():
    """Preprocess SST-2 dataset."""
    print("Processing SST-2 dataset...")
    
    sst2_path = Path('archive (5)/SST2-Data/SST2-Data/stanfordSentimentTreebank/stanfordSentimentTreebank')
    loader = SST2Loader(sst2_path)
    
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = loader.load()
    
    # Clean texts (SST-2 is already relatively clean, but do basic cleaning)
    print("Cleaning texts...")
    train_texts = [clean_text(text) for text in tqdm(train_texts)]
    val_texts = [clean_text(text) for text in tqdm(val_texts)]
    test_texts = [clean_text(text) for text in tqdm(test_texts)]
    
    # Save
    os.makedirs('data/sst2', exist_ok=True)
    
    # Train
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    train_df.to_csv('data/sst2/train.csv', index=False)
    
    # Val
    val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
    val_df.to_csv('data/sst2/val.csv', index=False)
    
    # Test
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    test_df.to_csv('data/sst2/test.csv', index=False)
    
    print(f"SST-2: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")


def preprocess_yelp(max_samples=None):
    """Preprocess Yelp dataset."""
    print("Processing Yelp dataset...")
    
    import json
    from utils.data_loader import YelpDataLoader
    
    yelp_path = Path('archive (7)/yelp_academic_dataset_review.json')
    if not yelp_path.exists():
        print(f"Yelp dataset not found at {yelp_path}. Skipping.")
        return
    
    loader = YelpDataLoader(yelp_path)
    texts, labels = loader.load(sample_size=max_samples, binary=True)
    
    # Clean texts
    print("Cleaning texts...")
    texts = [clean_text(text) for text in tqdm(texts)]
    
    # Split into train/val/test
    train_texts, test_texts, train_labels, test_labels = create_train_test_split(
        texts, labels, test_size=0.2
    )
    train_texts, val_texts, train_labels, val_labels = create_train_test_split(
        train_texts, train_labels, test_size=0.1
    )
    
    # Save
    os.makedirs('data/yelp', exist_ok=True)
    
    # Train
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    train_df.to_csv('data/yelp/train.csv', index=False)
    
    # Val
    val_df = pd.DataFrame({'text': val_texts, 'label': val_labels})
    val_df.to_csv('data/yelp/val.csv', index=False)
    
    # Test
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    test_df.to_csv('data/yelp/test.csv', index=False)
    
    print(f"Yelp: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")


def main():
    print("Preprocessing all datasets...")
    preprocess_imdb()
    preprocess_sst2()
    preprocess_yelp(max_samples=50000)  # Sample 50k reviews
    print("\nAll datasets preprocessed!")


if __name__ == '__main__':
    main()

