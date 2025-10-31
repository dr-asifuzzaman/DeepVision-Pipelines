# evaluate.py
import argparse, os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from config import MODEL_DIR, DETERMINISTIC
from data_loader import infer_structure, build_datasets
from train_utils import wrap_pipeline, collect_preds
from visualize import plot_cm, save_classification_report, plot_roc_all, prediction_gallery

def load_label_map(model_dir):
    with open(os.path.join(model_dir, "label_map.json")) as f:
        d = json.load(f)
    # keys are strings in JSON; sort by numeric key
    id_to_class = {int(k): v for k, v in d.items()}
    class_names = [id_to_class[i] for i in sorted(id_to_class)]
    return class_names, id_to_class

def load_saved_model(model_dir):
    sm_dir = os.path.join(model_dir, "best_savedmodel")
    print("Loading SavedModel:", sm_dir)
    try:
        return tf.keras.models.load_model(sm_dir)  # Keras 3 SavedModel (compiled)
    except Exception:
        return tf.saved_model.load(sm_dir)         # fallback (raw signature)

def evaluate(args):
    # dataset
    struct = infer_structure(args.image_dir, args.single_root)
    class_names, train_raw, val_raw, test_raw = build_datasets(struct)

    # model (compiled Keras if possible)
    model = load_saved_model(args.model_dir)

    # choose split
    eval_raw = test_raw if len(list(test_raw)) > 0 else val_raw
    opts = tf.data.Options(); opts.experimental_deterministic = DETERMINISTIC
    eval_raw = eval_raw.with_options(opts)
    eval_ds = wrap_pipeline(eval_raw)

    num_classes = len(class_names)
    y_true, y_pred, y_prob = collect_preds(model, eval_ds, num_classes)

    viz_dir = os.path.join(args.model_dir, "viz_eval")
    os.makedirs(viz_dir, exist_ok=True)

    # CM
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plot_cm(cm, class_names, "Confusion Matrix (Counts)", save_dir=viz_dir, fname="cm_counts.png")
    plot_cm(cm, class_names, "Confusion Matrix (Normalized)", save_dir=viz_dir, fname="cm_norm.png", normalize=True)

    # Report
    save_classification_report(y_true, y_pred, class_names, save_dir=viz_dir)

    # ROC
    try:
        y_true_oh = tf.one_hot(y_true, depth=num_classes).numpy()
        plot_roc_all(y_true_oh, y_prob, class_names, save_dir=viz_dir)
    except Exception as e:
        print("ROC–AUC skipped:", e)

    # Gallery
    id_to_class = {i:c for i,c in enumerate(class_names)}
    imgs, t, p, conf = [], [], [], []
    for xb, yb in eval_ds:
        probs = model.predict(xb, verbose=0)
        yhat  = np.argmax(probs, axis=1)
        cmax  = probs[np.arange(len(yhat)), yhat]
        for i in range(len(yhat)):
            if len(imgs) >= args.rows * args.cols: break
            imgs.append(xb[i].numpy().astype("uint8"))
            t.append(int(yb.numpy()[i])); p.append(int(yhat[i])); conf.append(float(cmax[i]))
        if len(imgs) >= args.rows * args.cols: break
    wrong = (np.array(t) != np.array(p))
    order = np.argsort(~wrong)

    prediction_gallery(np.array(imgs)[order], np.array(t)[order], np.array(p)[order],
                       np.array(conf)[order], wrong[order],
                       id_to_class, max_cols=args.cols, title="Prediction Gallery (Eval)",
                       save_dir=viz_dir, fname="prediction_gallery.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a saved model and generate visuals.")
    ap.add_argument("--image_dir", type=str, required=True, help="Root that has train/val/test or a single root.")
    ap.add_argument("--single_root", type=str, default=None)
    ap.add_argument("--model_dir", type=str, default=MODEL_DIR)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()
    evaluate(args)
