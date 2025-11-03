# visualize.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torchvision import utils

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import PLOTS_DIR


def _save_and_show(fig, filename: str):
    out = PLOTS_DIR / filename
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.show()  # blocking until the window is closed


def dataset_distribution(targets: np.ndarray, class_names: List[str], title: str, filename: str):
    counts = np.bincount(targets, minlength=len(class_names))
    fig = plt.figure(figsize=(8, 4))
    plt.bar(range(len(class_names)), counts)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.title(title)
    plt.tight_layout()
    _save_and_show(fig, filename)


def classwise_grid(image_paths: List[str], labels: List[int], class_names: List[str],
                   per_class: int = 6, filename: str = "classwise_grid.png"):
    """
    Shows a few examples per class (without transforms). image_paths should align with labels.
    """
    by_class = {i: [] for i in range(len(class_names))}
    for p, y in zip(image_paths, labels):
        if len(by_class[y]) < per_class:
            by_class[y].append(p)

    # Build grid: rows = classes, columns = per_class
    import PIL.Image as Image
    nrows = len(class_names)
    ncols = per_class
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2*ncols, 2.2*nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis('off')
            if c < len(by_class[r]):
                img = Image.open(by_class[r][c]).convert("RGB")
                ax.imshow(img)
        axes[r, 0].set_ylabel(class_names[r], rotation=0, labelpad=50, fontsize=10, va='center')
    fig.suptitle("Class-wise Sample Grid", y=0.99)
    plt.tight_layout()
    _save_and_show(fig, filename)


def batch_preview(loader, title="Batch Preview", filename="batch_preview.png", max_images=16):
    batch = next(iter(loader))
    imgs, labels = batch
    grid = utils.make_grid(imgs[:max_images], nrow=min(8, max_images), padding=2)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    _save_and_show(fig, filename)


def plot_training_curves(history: Dict[str, list], filename="training_curves.png"):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, history["train_loss"], label="Train")
    plt.plot(epochs_ran, history["val_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, history["train_acc"], label="Train")
    plt.plot(epochs_ran, history["val_acc"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()

    plt.tight_layout()
    _save_and_show(fig, filename)


def plot_confusion_matrices(y_true, y_pred, class_names, filename="confusion_matrices.png"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im0 = ax[0].imshow(cm, interpolation='nearest')
    ax[0].set_title("Confusion Matrix (Counts)")
    ax[0].set_xticks(range(len(class_names))); ax[0].set_xticklabels(class_names, rotation=45, ha="right")
    ax[0].set_yticks(range(len(class_names))); ax[0].set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(cm_norm, interpolation='nearest')
    ax[1].set_title("Confusion Matrix (Normalized)")
    ax[1].set_xticks(range(len(class_names))); ax[1].set_xticklabels(class_names, rotation=45, ha="right")
    ax[1].set_yticks(range(len(class_names))); ax[1].set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save_and_show(fig, filename)


def show_classification_report(y_true, y_pred, class_names, filename="classification_report.txt"):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    out = (PLOTS_DIR / filename)
    out.write_text(report)
    print(report)  # also print to console


def plot_roc_auc(y_true, probs, class_names, filename="roc_auc.png"):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1:
        # handle binary special-case
        y_bin = np.hstack([1 - y_bin, y_bin])
        probs = np.hstack([1 - probs, probs])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr["micro"], tpr["micro"], linestyle='--', label=f"micro (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--', label=f"macro (AUC={roc_auc['macro']:.3f})")
    for i, cls in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=':')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _save_and_show(fig, filename)


def prediction_gallery(imgs_tensor, y_true, y_pred, class_names, rows=2, cols=5, filename="prediction_gallery.png"):
    n = min(rows * cols, imgs_tensor.size(0))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2.8 * cols, 2.8 * rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = np.transpose(imgs_tensor[i].numpy(), (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')
        correct = (y_true[i] == y_pred[i])
        color = "green" if correct else "red"
        ax.set_title(f"T:{class_names[y_true[i]]}\nP:{class_names[y_pred[i]]}", color=color, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
    plt.suptitle("Prediction Gallery (Green=Correct, Red=Wrong)", y=1.01, fontsize=12)
    plt.tight_layout()
    _save_and_show(fig, filename)
