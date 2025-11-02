# Intermediate Data Directory

This directory stores preprocessed data that flows between notebooks.

## Data Flow

1. **Notebook 01** (`01_data_exploration.ipynb`): Explores raw datasets
2. **Notebook 02** (`02_preprocessing.ipynb`): Preprocesses and saves cleaned data here
3. **Notebooks 03+**: Load preprocessed data from this directory

## Files Generated

After running `02_preprocessing.ipynb`, you should have:
- `data/imdb_train_preprocessed.csv`
- `data/imdb_val_preprocessed.csv`
- `data/imdb_test_preprocessed.csv`
- `data/yelp_train_preprocessed.csv`
- `data/yelp_val_preprocessed.csv`
- `data/yelp_test_preprocessed.csv`
- `data/sst2_train_preprocessed.csv`
- `data/sst2_val_preprocessed.csv`
- `data/sst2_test_preprocessed.csv`

## Usage

```python
from src.data.dataset_loader import load_preprocessed_data

# Load preprocessed data (already split into train/val/test)
train_texts, train_labels = load_preprocessed_data('imdb_train', data_dir='../intermediate/data')
val_texts, val_labels = load_preprocessed_data('imdb_val', data_dir='../intermediate/data')
test_texts, test_labels = load_preprocessed_data('imdb_test', data_dir='../intermediate/data')
```

## Note

These files are ignored by git (in `.gitignore`) due to their size. 
Each team member should run `02_preprocessing.ipynb` to generate them locally.

