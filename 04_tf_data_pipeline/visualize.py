# visualize.py
import os, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay

from matplotlib.colors import LinearSegmentedColormap

# soft blush palette: white → peach → blush → rose
BLUSH_CMAP = LinearSegmentedColormap.from_list(
    "blush",
    ["#ffffff", "#ffe5e9", "#ffcbd5", "#ffb3c2", "#ff99ae", "#ff7a97", "#ff5c83"]
)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_distribution(class_names, counts, title, save_dir=None, fname="dist.png"):
    plt.figure()
    plt.bar(class_names, counts)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Samples")
    plt.tight_layout()
    if save_dir: 
        ensure_dir(save_dir); plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()

def show_batch(ds, id_to_class, title, max_images=16, save_dir=None, fname="grid.png"):
    images, labels = next(iter(ds))
    n = min(max_images, images.shape[0])
    cols = 4; rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*3, rows*3))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(id_to_class[int(labels[i].numpy())])
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    if save_dir:
        ensure_dir(save_dir); plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()

def visualize_augmentations(ds, augmenter, id_to_class, n=8, save_dir=None, fname="augment.png"):
    imgs, labels = next(iter(ds))
    imgs = imgs[:n]
    augmented = augmenter(imgs)
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        arr = np.clip(augmented[i].numpy(), 0, 255).astype("uint8")
        plt.imshow(arr)
        plt.title(id_to_class[int(labels[i].numpy())])
        plt.axis("off")
    plt.suptitle("Augmentation Preview")
    plt.tight_layout()
    if save_dir:
        ensure_dir(save_dir); plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()

def plot_history(history, save_dir=None):
    ensure_dir(save_dir or ".")
    hist = history.history if hasattr(history, "history") else history
    epochs = np.arange(1, len(hist.get("loss", [])) + 1)

    # Loss
    plt.figure(figsize=(7,4))
    plt.plot(epochs, hist["loss"], label="train_loss")
    if "val_loss" in hist: plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (Train vs Val)")
    plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150)
    plt.show()

    # Accuracy
    if "accuracy" in hist:
        plt.figure(figsize=(7,4))
        plt.plot(epochs, hist["accuracy"], label="train_acc")
        if "val_accuracy" in hist: plt.plot(epochs, hist["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (Train vs Val)")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(); plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, "accuracy.png"), dpi=150)
        plt.show()

def plot_cm(cm, class_names, title, save_dir=None, fname="cm.png", normalize=False):
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)

    plt.figure(figsize=(7,7))
    plt.imshow(mat, interpolation="nearest", cmap=BLUSH_CMAP)  # ★ blush
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    # slightly lower threshold on pinks so text stays legible
    thresh = np.nanmax(mat) * 0.55 if np.isfinite(mat).any() else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            # dark gray on light cells, white on dark cells
            color = "#1f1f1f" if val <= thresh else "white"
            plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()

def save_classification_report(y_true, y_pred, class_names, save_dir):
    ensure_dir(save_dir)
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification Report\n", report_txt)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

def plot_roc_all(y_true_oh, y_prob, class_names, save_dir=None, fname="roc_all_classes.png"):
    """
    Draws per-class ROC curves (solid), micro-average (purple dashed),
    and the diagonal baseline (brown dotted). Legend shows AUC per class.
    """
    os.makedirs(save_dir or ".", exist_ok=True)
    n_classes = y_true_oh.shape[1]

    # compute per-class curves + AUC
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i, cls in enumerate(class_names):
        # guard against classes absent in y_true
        if np.sum(y_true_oh[:, i]) == 0 or np.sum(1 - y_true_oh[:, i]) == 0:
            # cannot compute ROC; skip
            fpr_dict[i], tpr_dict[i], auc_dict[i] = np.array([0, 1]), np.array([0, 1]), float("nan")
            continue
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        try:
            auc_dict[i] = roc_auc_score(y_true_oh[:, i], y_prob[:, i])
        except Exception:
            auc_dict[i] = float("nan")

    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_oh.ravel(), y_prob.ravel())
    try:
        auc_micro = roc_auc_score(y_true_oh, y_prob, average="micro", multi_class="ovr")
    except Exception:
        auc_micro = float("nan")

    # figure
    plt.figure(figsize=(7, 7))

    # per-class (solid, default colors)
    for i, cls in enumerate(class_names):
        label_txt = f"{cls} (AUC={auc_dict[i]:.3f})" if np.isfinite(auc_dict[i]) else f"{cls} (AUC=NA)"
        plt.plot(fpr_dict[i], tpr_dict[i], label=label_txt, linewidth=2)

    # micro-average (purple dashed)
    plt.plot(
        fpr_micro, tpr_micro,
        linestyle="--", linewidth=2, color="#7B68EE",
        label=f"Micro-avg (AUC={auc_micro:.3f})" if np.isfinite(auc_micro) else "Micro-avg (AUC=NA)"
    )

    # diagonal baseline (brown dotted)
    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=2, color="#8B4513")

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (All Classes)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right", framealpha=0.9)
    out_path = os.path.join(save_dir or ".", fname)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.show()

    # also print macro AUC for convenience
    try:
        auc_macro = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
        print(f"Macro ROC–AUC: {auc_macro:.3f}")
    except Exception:
        pass

    return {"per_class": {class_names[i]: auc_dict[i] for i in range(n_classes)},
            "micro": auc_micro}

def prediction_gallery(imgs, t, p, conf, wrong, id_to_class, max_cols=4, title="Predictions", save_dir=None, fname="gallery.png"):
    n = len(imgs)
    cols = max_cols
    rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*3.2, rows*3.2))
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(imgs[i])
        ok = not wrong[i]
        color = "green" if ok else "red"
        ax.set_title(f"T: {id_to_class[int(t[i])]}  P: {id_to_class[int(p[i])]} ({conf[i]:.2f})", color=color, fontsize=10)
        ax.axis("off")
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2.5)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_dir:
        ensure_dir(save_dir); plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.show()
