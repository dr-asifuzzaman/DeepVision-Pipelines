# tf_record_pipeline/train_utils.py
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import LABELS_JSON, BEST_MODEL_PATH, FINAL_MODEL_PATH, TRAIN_LOG_CSV



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

def plot_history(history, history_ft=None):
    plt.figure(figsize=(10,4))
    # accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history.get("accuracy", []), label="Train")
    plt.plot(history.history.get("val_accuracy", []), label="Val")
    if history_ft is not None:
        plt.plot(history_ft.history.get("accuracy", []), label="Train (FT)")
        plt.plot(history_ft.history.get("val_accuracy", []), label="Val (FT)")
    plt.title("Accuracy"); plt.legend()
    # loss
    plt.subplot(1,2,2)
    plt.plot(history.history.get("loss", []), label="Train")
    plt.plot(history.history.get("val_loss", []), label="Val")
    if history_ft is not None:
        plt.plot(history_ft.history.get("loss", []), label="Train (FT)")
        plt.plot(history_ft.history.get("val_loss", []), label="Val (FT)")
    plt.title("Loss"); plt.legend()
    plt.tight_layout(); plt.show()

def export_final(model):
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FINAL_MODEL_PATH))
    print(f"Saved final model to: {FINAL_MODEL_PATH}")
