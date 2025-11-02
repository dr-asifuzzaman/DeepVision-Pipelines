# visualize.py
import os, io, math, datetime, random
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from itertools import cycle
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams["figure.dpi"] = 130


# Soft blush gradient (very light pink -> deeper rose)
BLUSH_CMAP = LinearSegmentedColormap.from_list(
    "blush",
    ["#fff0f3", "#ffd6e0", "#ffcad4", "#f4acb7", "#e27d9d", "#d66d75"]
)

def _tsdir(root: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, ts)
    os.makedirs(out, exist_ok=True)
    return out

def show_then_save(fig, save_path: str):
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    # Block until the figure window is closed
    plt.show()

def plot_distribution(dist: Dict[str, np.ndarray], classes: List[str], results_dir: str):
    outdir = _tsdir(os.path.join(results_dir, "dataset_distribution"))
    for split, counts in dist.items():
        fig = plt.figure(figsize=(8,4))
        x = np.arange(len(classes))
        plt.bar(x, counts)
        plt.xticks(x, classes, rotation=45, ha="right")
        plt.ylabel("Images")
        plt.title(f"{split.upper()} distribution")
        show_then_save(fig, os.path.join(outdir, f"{split}_distribution.png"))
        plt.close(fig)

def _sample_indices_per_class(y: List[int], classes: List[str], max_n: int):
    idx_by_class = {c: [] for c in range(len(classes))}
    for i, lab in enumerate(y):
        idx_by_class[lab].append(i)
    for k in idx_by_class:
        random.shuffle(idx_by_class[k])
        idx_by_class[k] = idx_by_class[k][:max_n]
    return idx_by_class

def classwise_grids(X: List[str], y: List[int], classes: List[str], rows: int, cols: int, results_dir: str):
    outdir = _tsdir(os.path.join(results_dir, "classwise_grids"))
    idx_by_class = _sample_indices_per_class(y, classes, rows*cols)
    import PIL.Image as Image

    for cidx, cname in enumerate(classes):
        sel = idx_by_class[cidx]
        if not sel: 
            continue
        fig = plt.figure(figsize=(cols*2.5, rows*2.5))
        fig.suptitle(f"Class: {cname}", y=0.98)
        for i, ix in enumerate(sel[:rows*cols]):
            ax = plt.subplot(rows, cols, i+1)
            ax.axis("off")
            try:
                img = Image.open(X[ix]).convert("RGB")
                img = img.resize((256,256))
                ax.imshow(img)
                ax.set_title(os.path.basename(X[ix])[:30], fontsize=8)
            except Exception as e:
                ax.text(0.5,0.5,str(e),ha="center",va="center",fontsize=8)
        savep = os.path.join(outdir, f"class_{cname}.png")
        show_then_save(fig, savep)
        plt.close(fig)

def augmentation_preview(sample_paths: List[str], augment_layer, results_dir: str, repeats: int = 6):
    if augment_layer is None or len(sample_paths) == 0:
        return
    import tensorflow as tf
    outdir = _tsdir(os.path.join(results_dir, "augmentation_previews"))
    # take 1–2 random images
    picks = random.sample(sample_paths, k=min(2, len(sample_paths)))
    for p in picks:
        raw = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        raw = tf.image.convert_image_dtype(raw, tf.float32)
        raw = tf.image.resize(raw, (224,224))
        imgs = [augment_layer(tf.expand_dims(raw,0), training=True)[0].numpy() for _ in range(repeats)]
        fig = plt.figure(figsize=(repeats*2.2, 2.5))
        for i, im in enumerate(imgs):
            ax = plt.subplot(1, repeats, i+1)
            ax.axis("off")
            ax.imshow(np.clip(im,0,1))
        savep = os.path.join(outdir, f"aug_{os.path.basename(p)}.png")
        show_then_save(fig, savep)
        plt.close(fig)

def plot_history(history, results_dir: str):
    outdir = _tsdir(os.path.join(results_dir, "training_curves"))
    hist = history.history
    # Loss
    fig = plt.figure(figsize=(6,4))
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend()
    show_then_save(fig, os.path.join(outdir, "loss.png")); plt.close(fig)

    # Accuracy (if present)
    for m in ("accuracy","acc","categorical_accuracy"):
        if m in hist or f"val_{m}" in hist:
            fig2 = plt.figure(figsize=(6,4))
            plt.plot(hist.get(m, []), label=f"train_{m}")
            plt.plot(hist.get(f"val_{m}", []), label=f"val_{m}")
            plt.xlabel("Epoch"); plt.ylabel(m); plt.title(m.capitalize())
            plt.legend()
            show_then_save(fig2, os.path.join(outdir, f"{m}.png")); plt.close(fig2)
            break

def plot_confusion_matrices(y_true, y_pred, classes: List[str], results_dir: str):
    outdir = _tsdir(os.path.join(results_dir, "confusion_matrices"))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    # -------- Raw counts ----------
    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=BLUSH_CMAP)
    plt.title("Confusion Matrix (Counts)")
    plt.colorbar(im)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    threshold = cm.max() * 0.5 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            txt_color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, val, ha="center", va="center", fontsize=8, color=txt_color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    show_then_save(fig, os.path.join(outdir, "confusion_counts.png"))
    plt.close(fig)

    # -------- Normalized ----------
    with np.errstate(all="ignore"):
        cmn = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)

    fig2 = plt.figure(figsize=(6, 5))
    im2 = plt.imshow(cmn, interpolation="nearest", cmap=BLUSH_CMAP, vmin=0.0, vmax=1.0)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar(im2)
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            val = f"{cmn[i, j]:.2f}"
            txt_color = "white" if cmn[i, j] > 0.5 else "black"
            plt.text(j, i, val, ha="center", va="center", fontsize=8, color=txt_color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    show_then_save(fig2, os.path.join(outdir, "confusion_normalized.png"))
    plt.close(fig2)

def plot_classification_report(y_true, y_pred, classes: List[str], results_dir: str):
    outdir = _tsdir(os.path.join(results_dir, "classification_report"))
    rpt = classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0)
    # Render as figure
    fig = plt.figure(figsize=(10, 0.5*len(classes) + 3))
    plt.axis("off")
    plt.title("Classification Report", pad=20)
    plt.text(0.01, 0.99, rpt, ha="left", va="top", family="monospace")
    show_then_save(fig, os.path.join(outdir, "classification_report.png"))
    plt.close(fig)

def plot_roc(y_true_onehot: np.ndarray, y_prob: np.ndarray, classes: List[str], results_dir: str):
    outdir = _tsdir(os.path.join(results_dir, "roc_auc"))
    n_classes = len(classes)
    # Per-class ROC
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    fig = plt.figure(figsize=(7,6))
    plt.plot(fpr["micro"], tpr["micro"], linestyle=':', label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(all_fpr, mean_tpr, linestyle=':', label=f"macro-average (AUC = {roc_auc['macro']:.3f})")
    colors = cycle(range(n_classes))
    for i, _c in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8, loc="lower right", ncol=1)
    show_then_save(fig, os.path.join(outdir, "roc_curves.png"))
    plt.close(fig)

def prediction_gallery(filepaths: List[str], 
                       y_true: List[int], 
                       y_pred: List[int], 
                       y_prob: np.ndarray,        # <— add probs
                       classes: List[str],
                       rows: int, cols: int, 
                       results_dir: str):
    import PIL.Image as Image
    outdir = _tsdir(os.path.join(results_dir, "prediction_gallery"))
    n = rows * cols
    idxs = list(range(len(filepaths)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    fig = plt.figure(figsize=(cols * 2.6, rows * 2.6))
    for i, ix in enumerate(idxs):
        ax = plt.subplot(rows, cols, i + 1)
        ax.axis("off")
        try:
            img = Image.open(filepaths[ix]).convert("RGB").resize((256, 256))
            ok = (y_true[ix] == y_pred[ix])
            conf = float(y_prob[ix, y_pred[ix]]) if y_prob is not None else float("nan")

            ax.imshow(img)

            # Title with colored text + confidence %
            t_lbl = classes[y_true[ix]]
            p_lbl = classes[y_pred[ix]]
            title = f"T: {t_lbl} | P: {p_lbl} ({conf*100:.1f}%)"
            ax.set_title(title, fontsize=9, color=("green" if ok else "red"))

            # Keep the green/red border too
            for spine in ax.spines.values():
                spine.set_linewidth(4.0)
                spine.set_color("green" if ok else "red")
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", fontsize=8)
    show_then_save(fig, os.path.join(outdir, "prediction_gallery.png"))
    plt.close(fig)
