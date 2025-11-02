# config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    # I/O
    image_dir: str = ""
    artifacts_dir: str = "model_artifacts"
    results_dir: str = "results"

    # Data
    img_size: tuple = (224, 224)
    channels: int = 3
    batch_size: int = 32
    seed: int = 1337
    val_split: float = 0.10
    test_split: float = 0.10
    shuffle_buffer: int = 2048
    cache_dataset: bool = True
    prefetch_batches: int = 2

    # Model/Train
    base_learning_rate: float = 1e-4
    epochs: int = 10
    label_smoothing: float = 0.0
    augment: bool = True
    class_weighting: bool = True
    early_stop_patience: int = 15
    reduce_lr_patience: int = 20
    monitor_metric: str = "val_loss"

    # Visualization
    fig_dpi: int = 130
    grid_rows: int = 2
    grid_cols: int = 5
    max_samples_per_class_preview: int = 8

def ensure_dirs(cfg: Config):
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
