# Sentiment Analysis Project

Deep Learning approaches for Text Sentiment Analysis (TSA) with a focus on:
1. **Domain Adaptation**: Adapting networks trained on one domain to work in another
2. **Transformer Architectures**: Comparing different Transformer models (BERT, RoBERTa, XLNet, ELECTRA)
3. **Small Dataset Scenarios**: Techniques for handling insufficient training samples

## Current Status

✅ Project structure created  
✅ Conda environment setup  
✅ Datasets preprocessed (IMDB: 50k, SST-2: 11.8k, Yelp: 50k samples)  
✅ Baseline models implemented (LSTM, CNN, BiLSTM+Attention)  
✅ Transformer models implemented (BERT, RoBERTa, XLNet, ELECTRA)  
✅ Training scripts ready  
🔄 Domain adaptation techniques (in progress)  
🔄 Model training and evaluation (pending)  
🔄 Final report (pending)

## Datasets
- Stanford Sentiment Treebank (SST-2)
- IMDB Movie Reviews
- Yelp Reviews

## Setup

Using Conda (recommended):
```bash
conda env create -f environment.yaml
conda activate sentiment-analysis
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Quick Start

Preprocess datasets:
```bash
python scripts/preprocess_datasets.py
```

Train a model:
```bash
python scripts/train.py --model_type bert --model_name bert-base-uncased
```

## Project Structure

```
.
├── data/              # Processed datasets
├── models/            # Model implementations
├── notebooks/         # Jupyter notebooks for exploration
├── scripts/           # Training and evaluation scripts
├── utils/             # Utility functions
└── results/           # Results and outputs
```

## References

1. Gui, T. et al. (2019). Long Short-Term Memory with Dynamic Skip Connections. AAAI Conf. Artif. Intell., 33(01), 6481-6488.
2. Maas, A. L. et al. (2011). Learning word vectors for sentiment analysis. ACL-HLT 2011.
3. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. Advances in Neural Information Processing Systems.

