"""
Configuration loader.
"""
import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, config_path: str = "config.yaml"):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

