# visualize.py (drop-in replacement)
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import os, json, random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from config import IMG_SIZE, CHANNELS, LABELS_JSON, ARTIFACTS_DIR

PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _can_show_gui() -> bool:
    """
    Returns True if a GUI backend is likely available.
    On WSL/Linux, requires DISPLAY. On headless it returns False.
    """
    # If user forced non-interactive backend
    backend = matplotlib.get_backend().lower()
    non_gui_backends = {"agg", "cairoagg", "pdf", "svg", "ps", "template"}
    if backend in non_gui_backends:
        return False
    # On Linux/WSL must have DISPLAY
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        return False
    return True

def _smart_show(fig: plt.Figure, fname: str):
    """
    Save figure to PLOTS_DIR/fname. If GUI is available, also show (blocking).
    """
    out = PLOTS_DIR / fname
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[PLOT] Saved: {out}")
    if _can_show_gui():
        plt.show(block=True)
    else:
        # No GUI—close to free memory
        plt.close(fig)

# ---------- Dataset distribution ----------
def plot_split_distribution(train_items, val_items, test_items, classes: List[str]):
    if train_items is None:  # tfrecords pre-existing; no counts we can show
        print("Skip dataset distribution (reused existing TFRecords).")
        return
    def count_per(items):
        c = np.zeros(len(classes), int)
        for _, lab in items: c[lab]+=1
        return c
    train_c, val_c, test_c = count_per(train_items), count_per(val_items), count_per(test_items)
    x = np.arange(len(classes)); w = 0.25
    fig = plt.figure(figsize=(10,4))
    plt.bar(x-w, train_c, width=w, label="Train")
    plt.bar(x,   val_c,   width=w, label="Val")
    plt.bar(x+w, test_c,  width=w, label="Test")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Dataset Distribution by Class")
    plt.legend()
    plt.tight_layout()
    _smart_show(fig, "01_dataset_distribution.png")

# ---------- Class-wise grids ----------
def classwise_grid(train_items: List[Tuple[str,int]], classes: List[str], k_per_class=5):
    if not train_items:
        print("Skip class-wise grid (no training items known).")
        return
    paths = defaultdict(list)
    for p, lab in train_items:
        if len(paths[lab]) < k_per_class:
            paths[lab].append(p)

    rows = len(classes); cols = k_per_class
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axs = np.atleast_2d(axs)
    for r, cname in enumerate(classes):
        for c in range(cols):
            ax = axs[r, c]; ax.axis("off")
            try:
                img = Image.open(paths[r][c]).convert("RGB").resize(IMG_SIZE[::-1])
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            if c == 0:
                ax.set_title(cname)
    plt.suptitle("Class-wise Image Grid", y=0.99)
    plt.tight_layout()
    _smart_show(fig, "02_classwise_grid.png")

# ---------- Augmentation previews ----------
def _augment_demo(img: tf.Tensor):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

def _load_img_tensor(path: str):
    t = tf.io.decode_image(tf.io.read_file(path), channels=CHANNELS, expand_animations=False)
    t = tf.image.convert_image_dtype(t, tf.float32)
    t = tf.image.resize(t, IMG_SIZE, antialias=True)
    return t

def show_augmentation_preview(train_items, aug_samples=6):
    if not train_items:
        print("Skip augmentation preview (no training items known).")
        return
    base_path = train_items[0][0]
    base = _load_img_tensor(base_path)
    cols = aug_samples + 1
    fig, axs = plt.subplots(1, cols, figsize=(3*cols, 3))
    axs[0].imshow(tf.clip_by_value(base, 0, 1)); axs[0].set_title("Original"); axs[0].axis("off")
    for i in range(aug_samples):
        a = _augment_demo(base)
        axs[i+1].imshow(tf.clip_by_value(a,0,1)); axs[i+1].set_title(f"Aug {i+1}"); axs[i+1].axis("off")
    plt.suptitle("Augmentation Previews")
    plt.tight_layout()
    _smart_show(fig, "03_augmentation_previews.png")

# ---------- Prediction gallery ----------
def prediction_gallery(model, items: List[Tuple[str,int]], classes: List[str],
                       rows=2, cols=5, fname: str = "99_prediction_gallery.png"):
    n = rows * cols
    subset = items[:n]
    import numpy as np
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    axs = np.atleast_2d(axs)
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    for i, (path, true_lab) in enumerate(subset):
        r, c = divmod(i, cols); ax = axs[r, c]; ax.axis("off")
        img = Image.open(path).convert("RGB").resize(IMG_SIZE[::-1])
        t = tf.image.convert_image_dtype(np.array(img), tf.float32)[None, ...]
        t = preprocess(t*255.0)
        probs = model.predict(t, verbose=0)[0]
        pred = int(np.argmax(probs)); conf = float(probs[pred])
        ok = (pred == true_lab)
        tlabel = classes[true_lab] if 0 <= true_lab < len(classes) else "Unknown"
        title = f"P:{classes[pred]} ({conf*100:.1f}%) | T:{tlabel}"
        ax.set_title(title, color=("green" if ok else "red"), fontsize=9)
        ax.imshow(img)
    for j in range(i+1, rows*cols): axs.flat[j].axis("off")
    plt.suptitle("Prediction Gallery (green=correct, red=wrong)", y=0.99)
    plt.tight_layout()
    _smart_show(fig, fname)
