# predict.py
import argparse, os, random, numpy as np, tensorflow as tf
from config import Config, ensure_dirs
from data_loader import load_datasets
from visualize import prediction_gallery

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True, type=str)
    p.add_argument("--rows", default=2, type=int)
    p.add_argument("--cols", default=5, type=int)
    return p.parse_args()

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
    return y_true_idx, y_pred

def main():
    args = parse_args()
    cfg = Config(image_dir=args.image_dir)
    ensure_dirs(cfg)

    _, _, test_ds, classes, _, index = load_datasets(cfg)

    best_path = os.path.join(cfg.artifacts_dir, "best_model.keras")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"Saved model not found at {best_path}. Train first with main.py")

    model = tf.keras.models.load_model(best_path)

    y_true, y_pred = collect_preds(model, test_ds)

    X_test, _ = index["test"]
    prediction_gallery(X_test, y_true, y_pred, classes, rows=args.rows, cols=args.cols, results_dir=cfg.results_dir)
    print("Prediction gallery shown and saved in results/")

if __name__ == "__main__":
    main()
