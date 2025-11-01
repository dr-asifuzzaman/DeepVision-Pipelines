import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import LABELS_JSON, IMG_SIZE, CHANNELS, VAL_SPLIT, TEST_SPLIT, FINAL_MODEL_PATH, BEST_MODEL_PATH, ARTIFACTS_DIR
from data_loader import make_dataset, list_classes, ensure_tfrecords, list_images_with_labels
from visualize import prediction_gallery  # now accepts fname
# use the same save/show behavior as other plots
PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _can_show_gui() -> bool:
    backend = matplotlib.get_backend().lower()
    if backend in {"agg","cairoagg","pdf","svg","ps","template"}:
        return False
    import os
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

def _load_model():
    if BEST_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(BEST_MODEL_PATH))
    return tf.keras.models.load_model(str(FINAL_MODEL_PATH))

def _plot_confusion_matrices(y_true, y_pred, classes):
    # counts
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Counts)")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    vmax = cm.max()
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            plt.text(j, i, val, ha="center", va="center",
                     color=("white" if val > vmax/2 else "black"))
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    _smart_show(fig, "05_confusion_counts.png")

    # normalized
    cmn = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cmn, cmap="Blues")
    plt.title("Confusion Matrix (Normalized)")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cmn[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=("white" if val > 0.5 else "black"))
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    _smart_show(fig, "06_confusion_normalized.png")

def _plot_roc(y_true, y_prob, classes):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    macro_auc = float(np.mean([roc_auc[i] for i in range(len(classes))]))

    fig = plt.figure(figsize=(7,6))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=1, label=f"{classes[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["micro"], tpr["micro"], "--", lw=2, label=f"Micro (AUC={roc_auc['micro']:.2f})")
    plt.plot([0,1],[0,1],":")
    plt.title(f"ROC Curves (macro AUC={macro_auc:.2f})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    _smart_show(fig, "07_roc_curves.png")

def main():
    ap = argparse.ArgumentParser(description="Evaluate saved model on TFRecord test split")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    classes = list_classes(args.image_dir)  # ensure consistent order
    ensure_tfrecords(args.image_dir, classes, VAL_SPLIT, TEST_SPLIT)  # no-op if exists

    test_ds = make_dataset("test", batch_size=32, training=False)
    model = _load_model()
    metrics = model.evaluate(test_ds, return_dict=True)
    print("Test metrics:", metrics)

    # Gather predictions
    y_true, y_pred, y_prob = [], [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(p, axis=1).tolist())
        y_prob.extend(p.tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)

    # 5 & 6: Confusion matrices
    _plot_confusion_matrices(y_true, y_pred, classes)

    # 6: Classification report (printed)
    print("\nClassification Report")
    print("=====================")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    # 7: ROC curves
    _plot_roc(y_true, y_prob, classes)

    # 8: Prediction gallery (deterministic sample, saved as #8)
    all_items = list_images_with_labels(args.image_dir, classes)
    random.Random(42).shuffle(all_items)
    n = args.rows * args.cols
    subset = all_items[:n]
    # Try to infer true label by folder name to color green/red
    inferred = []
    for p, _ in subset:
        parent = Path(p).parent.name
        tlabel = classes.index(parent) if parent in classes else 0
        inferred.append((p, tlabel))
    prediction_gallery(model, inferred, classes, rows=args.rows, cols=args.cols,
                       fname="08_prediction_gallery.png")

if __name__ == "__main__":
    main()
