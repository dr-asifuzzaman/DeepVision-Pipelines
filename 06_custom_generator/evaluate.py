# evaluate.py
import argparse, os, numpy as np, tensorflow as tf
from config import Config, ensure_dirs
from data_loader import load_datasets
from visualize import (
    plot_confusion_matrices, plot_classification_report, plot_roc, prediction_gallery
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True, type=str)
    p.add_argument("--rows", default=2, type=int)
    p.add_argument("--cols", default=5, type=int)
    return p.parse_args()

# def collect_preds(model, ds):
#     y_true = []; y_prob = []
#     for xb, yb in ds:
#         pb = model.predict(xb, verbose=0)
#         y_prob.append(pb)
#         y_true.append(yb.numpy())
#     y_prob = np.concatenate(y_prob, axis=0)
#     y_true = np.concatenate(y_true, axis=0)
#     y_pred = np.argmax(y_prob, axis=1)
#     y_true_idx = np.argmax(y_true, axis=1)
#     return y_true_idx, y_pred, y_prob, y_true


def collect_preds(model, ds):
    y_true = []; y_prob = []
    for xb, yb in ds:
        pb = model.predict(xb, verbose=0)
        y_prob.append(pb)
        y_true.append(yb.numpy())
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)
    return y_true_idx, y_pred, y_prob





def main():
    args = parse_args()
    cfg = Config(image_dir=args.image_dir)
    ensure_dirs(cfg)

    _, _, test_ds, classes, _, index = load_datasets(cfg)

    best_path = os.path.join(cfg.artifacts_dir, "best_model.keras")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Saved model not found at {best_path}. Train first with main.py")

    model = tf.keras.models.load_model(best_path)

    # y_true, y_pred, y_prob, y_true_onehot = collect_preds(model, test_ds)

    y_true, y_pred, y_prob = collect_preds(model, test_ds)

    # Plots (each will pop and save)
    plot_confusion_matrices(y_true, y_pred, classes, cfg.results_dir)
    plot_classification_report(y_true, y_pred, classes, cfg.results_dir)
    plot_roc(y_true_onehot, y_prob, classes, cfg.results_dir)

    # Gallery
    X_test, _ = index["test"]
    # prediction_gallery(X_test, y_true, y_pred, classes, rows=args.rows, cols=args.cols, results_dir=cfg.results_dir)

    prediction_gallery(
        X_test, y_true, y_pred, y_prob, classes,
        rows=args.rows, cols=args.cols, results_dir=cfg.results_dir
    )
    print("Evaluation complete. Plots saved under:", cfg.results_dir)

if __name__ == "__main__":
    main()
