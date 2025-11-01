# tf_record_pipeline/evaluate.py
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import LABELS_JSON, IMG_SIZE, CHANNELS
from data_loader import make_dataset, list_classes, ensure_tfrecords
from visualize import prediction_gallery
from config import VAL_SPLIT, TEST_SPLIT
from config import FINAL_MODEL_PATH, BEST_MODEL_PATH


from data_loader import list_images_with_labels
import numpy as np, random
from pathlib import Path

all_items = list_images_with_labels(args.image_dir, classes)
if not all_items:
    print("[WARN] No images found for gallery.")
    return

# Use a fixed seed so sampling is deterministic and reproducible
rng = random.Random(42)
rng.shuffle(all_items)

# Try to infer true labels from folder names to color titles
inferred_items = []
for p, _ in all_items:
    parent = Path(p).parent.name
    if parent in classes:
        inferred_items.append((p, classes.index(parent)))
    else:
        # unknown true label; mark class 0 so it will likely show as red
        inferred_items.append((p, 0))

# Choose exactly rows*cols images
n = args.rows * args.cols
subset = inferred_items[:n]
print(f"[Gallery] Showing {len(subset)} images (requested {n}).")

from visualize import prediction_gallery
prediction_gallery(model, subset, classes, rows=args.rows, cols=args.cols)







def _load_classes():
    with open(LABELS_JSON) as f:
        return json.load(f)["classes"]

def _load_model():
    if BEST_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(BEST_MODEL_PATH))
    return tf.keras.models.load_model(str(FINAL_MODEL_PATH))

def _plot_confusion_matrices(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    # counts
    plt.figure(figsize=(6,5))
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
    plt.tight_layout(); plt.show()

    # normalized
    cmn = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6,5))
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
    plt.tight_layout(); plt.show()

def _plot_roc(y_true, y_prob, classes):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # macro average
    all_aucs = list(roc_auc[i] for i in range(len(classes)))
    macro_auc = float(np.mean(all_aucs))

    plt.figure(figsize=(7,6))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=1, label=f"{classes[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(fpr["micro"], tpr["micro"], "--", lw=2, label=f"Micro (AUC={roc_auc['micro']:.2f})")
    plt.plot([0,1],[0,1],":")
    plt.title(f"ROC Curves (macro AUC={macro_auc:.2f})")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout(); plt.show()

def main():
    ap = argparse.ArgumentParser(description="Evaluate saved model on TFRecord test split")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    classes = list_classes(args.image_dir)  # ensures order equals training
    ensure_tfrecords(args.image_dir, classes, VAL_SPLIT, TEST_SPLIT)  # no-op if already exists

    test_ds = make_dataset("test", batch_size=32, training=False)
    model = _load_model()
    metrics = model.evaluate(test_ds, return_dict=True)
    print("Test metrics:", metrics)

    # Collect preds
    y_true, y_pred, y_prob = [], [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(p, axis=1).tolist())
        y_prob.extend(p.tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)

    # Confusion matrices
    _plot_confusion_matrices(y_true, y_pred, classes)

    # Classification report
    print("\nClassification Report")
    print("=====================")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    # ROC curves
    _plot_roc(y_true, y_prob, classes)

    # Optional prediction gallery on test items (pull file paths from labels/tfrecords not stored)
    # We’ll reuse prediction_gallery by sampling from the training image folders:
    from .data_loader import list_images_with_labels, split_items
    all_items = list_images_with_labels(args.image_dir, classes)
    _, _, test_items = split_items(all_items, val_split=0.15, test_split=0.15)
    from .visualize import prediction_gallery
    prediction_gallery(model, test_items, classes, rows=args.rows, cols=args.cols)






if __name__ == "__main__":
    from sklearn.metrics import classification_report  # placed here to avoid top-level heavy import if unused
    main()
