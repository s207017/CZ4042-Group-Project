"""
Logging utilities for experiments.
Supports wandb and mlflow integration.
"""
import logging
import os
from typing import Optional, Dict
from pathlib import Path


def setup_logger(name: str = "sentiment_analysis", 
                log_dir: str = "logs",
                level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir_path / f"{name}.log")
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


class WandbLogger:
    """Wrapper for Weights & Biases logging."""
    
    def __init__(self, project: str = "sentiment-analysis", enabled: bool = False):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            enabled: Whether to enable WandB
        """
        self.enabled = enabled
        self.project = project
        
        if enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("wandb not installed. Install with: pip install wandb")
                self.enabled = False
    
    def init(self, config: Dict, name: Optional[str] = None):
        """Initialize WandB run."""
        if self.enabled:
            self.wandb.init(project=self.project, config=config, name=name)
    
    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled:
            self.wandb.finish()


class MLflowLogger:
    """Wrapper for MLflow logging."""
    
    def __init__(self, experiment_name: str = "sentiment-analysis", enabled: bool = False):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: MLflow experiment name
            enabled: Whether to enable MLflow
        """
        self.enabled = enabled
        self.experiment_name = experiment_name
        
        if enabled:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_experiment(experiment_name)
            except ImportError:
                print("mlflow not installed. Install with: pip install mlflow")
                self.enabled = False
    
    def start_run(self, run_name: Optional[str] = None):
        """Start MLflow run."""
        if self.enabled:
            self.mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict):
        """Log parameters."""
        if self.enabled:
            self.mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            self.mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model."""
        if self.enabled:
            self.mlflow.pytorch.log_model(model, artifact_path)
    
    def end_run(self):
        """End MLflow run."""
        if self.enabled:
            self.mlflow.end_run()

