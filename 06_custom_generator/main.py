# main.py
import argparse, os, numpy as np, tensorflow as tf
from config import Config, ensure_dirs
from data_loader import load_datasets, get_augmentation
from model_builder import build_model
from train_utils import get_callbacks, compute_weights, save_class_index
from visualize import (
    plot_distribution, classwise_grids, augmentation_preview, plot_history,
    plot_confusion_matrices, plot_classification_report, plot_roc, prediction_gallery
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True, type=str)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--augment", default=1, type=int, help="1=on, 0=off")
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
    return y_true_idx, y_pred, y_prob, y_true

def main():
    args = parse_args()
    cfg = Config(
        image_dir=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=bool(args.augment),
    )
    ensure_dirs(cfg)

    # Load datasets & index
    train_ds, val_ds, test_ds, classes, dist, index = load_datasets(cfg)
    save_class_index(classes, cfg)

    # 1) Dataset distribution graphs
    plot_distribution(dist, classes, cfg.results_dir)

    # 2) Class-wise grids (from train index)
    X_train, y_train = index["train"]
    classwise_grids(X_train, y_train, classes, rows=cfg.grid_rows, cols=cfg.grid_cols, results_dir=cfg.results_dir)

    # 3) Augmentation previews
    augment_layer = get_augmentation(cfg) if cfg.augment else None
    augmentation_preview(X_train, augment_layer, cfg.results_dir, repeats=6)

    # Build model
    model = build_model(cfg, num_classes=len(classes))

    # Class weights
    class_weight = None
    if cfg.class_weighting:
        class_weight = compute_weights(np.array(y_train), num_classes=len(classes))

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=get_callbacks(cfg),
        class_weight=class_weight,
        verbose=1
    )

    # Curves
    plot_history(history, cfg.results_dir)

    # Evaluate on test for final visuals
    best_path = os.path.join(cfg.artifacts_dir, "best_model.keras")
    if os.path.isfile(best_path):
        model = tf.keras.models.load_model(best_path)

    y_true, y_pred, y_prob, y_true_onehot = collect_preds(model, test_ds)

    # Confusion matrices
    plot_confusion_matrices(y_true, y_pred, classes, cfg.results_dir)

    # Classification report
    plot_classification_report(y_true, y_pred, classes, cfg.results_dir)

    # ROC / AUC
    plot_roc(y_true_onehot, y_prob, classes, cfg.results_dir)

    # Prediction gallery (correct green, wrong red)
    X_test, y_test = index["test"]
    # prediction_gallery(X_test, y_true, y_pred, classes, rows=args.rows, cols=args.cols, results_dir=cfg.results_dir)

    prediction_gallery(
        X_test, y_true, y_pred, y_prob, classes,
        rows=args.rows, cols=args.cols, results_dir=cfg.results_dir
    )    

    print("\nAll figures were shown sequentially and also saved under:", cfg.results_dir)
    print("Best model saved to:", best_path)

if __name__ == "__main__":
    main()
