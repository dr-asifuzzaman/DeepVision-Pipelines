import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from config import Config, ensure_dirs

def get_callbacks(cfg: Config):
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.artifacts_dir, "best_model.keras"),
        monitor=cfg.monitor_metric,
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor=cfg.monitor_metric, mode="min",
        patience=cfg.early_stop_patience, restore_best_weights=True, verbose=1
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=cfg.monitor_metric, mode="min",
        patience=cfg.reduce_lr_patience, factor=0.5, verbose=1
    )
    return [ckpt, es, rlrop]

def compute_weights(y_train: np.ndarray, num_classes: int):
    classes = np.arange(num_classes)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {i: float(wi) for i, wi in enumerate(w)}

def save_class_index(classes, cfg: Config):
    import json
    path = os.path.join(cfg.artifacts_dir, "class_index.json")
    with open(path, "w") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)
    return path
