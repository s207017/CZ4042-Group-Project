"""
Set random seeds for reproducibility.
"""
import random
import numpy as np
import torch
import os


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

