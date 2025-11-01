# train_utils.py (drop-in replacement)
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    LABELS_JSON, BEST_MODEL_PATH, FINAL_MODEL_PATH,
    TRAIN_LOG_CSV, ARTIFACTS_DIR
)

PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _can_show_gui() -> bool:
    backend = matplotlib.get_backend().lower()
    non_gui = {"agg", "cairoagg", "pdf", "svg", "ps", "template"}
    if backend in non_gui:
        return False
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        return False
    return True

def _smart_show(fig: plt.Figure, fname: str):
    out = PLOTS_DIR / fname
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[PLOT] Saved: {out}")
    if _can_show_gui():
        plt.show(block=True)
    else:
        plt.close(fig)

def save_labels(classes):
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_JSON, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

def load_labels():
    with open(LABELS_JSON) as f:
        return json.load(f)["classes"]

def compute_weights_from_items(train_items, num_classes):
    if not train_items:
        return None
    y = np.array([lab for _, lab in train_items])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y)
    return {i: float(w) for i, w in enumerate(weights)}

def callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(str(BEST_MODEL_PATH), monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(TRAIN_LOG_CSV))
    ]

def _plot_single(ax, h, key, label):
    vals = h.history.get(key, [])
    if vals:
        ax.plot(vals, label=label)

def plot_history(history, history_ft=None):
    """Save + (optionally) show training/validation curves."""
    fig = plt.figure(figsize=(10, 4))

    # Accuracy subplot
    ax1 = plt.subplot(1, 2, 1)
    _plot_single(ax1, history, "accuracy", "Train")
    _plot_single(ax1, history, "val_accuracy", "Val")
    if history_ft is not None:
        _plot_single(ax1, history_ft, "accuracy", "Train (FT)")
        _plot_single(ax1, history_ft, "val_accuracy", "Val (FT)")
    ax1.set_title("Accuracy"); ax1.legend()

    # Loss subplot
    ax2 = plt.subplot(1, 2, 2)
    _plot_single(ax2, history, "loss", "Train")
    _plot_single(ax2, history, "val_loss", "Val")
    if history_ft is not None:
        _plot_single(ax2, history_ft, "loss", "Train (FT)")
        _plot_single(ax2, history_ft, "val_loss", "Val (FT)")
    ax2.set_title("Loss"); ax2.legend()

    plt.tight_layout()
    _smart_show(fig, "04_train_val_curves.png")

def export_final(model):
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FINAL_MODEL_PATH))
    print(f"Saved final model to: {FINAL_MODEL_PATH}")
