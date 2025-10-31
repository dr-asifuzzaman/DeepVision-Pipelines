# main.py
import argparse, os, json
import numpy as np
import tensorflow as tf

from config import *
from data_loader import infer_structure, build_datasets
from model_builder import build_model
from train_utils import count_by_class, wrap_pipeline, callbacks, collect_preds, export_model, save_history_csv
from visualize import (
    plot_distribution, show_batch, visualize_augmentations,
    plot_history, plot_cm, save_classification_report, plot_roc_all, prediction_gallery
)

from sklearn.metrics import confusion_matrix


def get_augmenter(enabled: bool):
    if not enabled:
        return tf.keras.Sequential([], name="no_augmentation")
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")

def main(args):
    # Override config by CLI
    data_root = args.image_dir or DATA_ROOT
    single_root = None if args.single_root is None else args.single_root
    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH_SIZE
    augment = bool(args.augment)

    if args.tfdata_path:
        print("[Info] --tfdata_path provided but not used in this pipeline.")

    # Build datasets
    struct = infer_structure(data_root, single_root)
    class_names, train_raw, val_raw, test_raw = build_datasets(struct)

    # Rebatch if user changed batch size
    if batch_size != BATCH_SIZE:
        train_raw = train_raw.unbatch().batch(batch_size)
        val_raw   = val_raw.unbatch().batch(batch_size)
        test_raw  = test_raw.unbatch().batch(batch_size)

    num_classes = len(class_names)
    id_to_class = {i:c for i,c in enumerate(class_names)}

    # Determinism option
    opts = tf.data.Options(); opts.experimental_deterministic = DETERMINISTIC
    train_raw = train_raw.with_options(opts); val_raw = val_raw.with_options(opts); test_raw = test_raw.with_options(opts)

    # Counts and wrapped pipelines
    train_counts = count_by_class(train_raw, num_classes)
    val_counts   = count_by_class(val_raw, num_classes)
    test_counts  = count_by_class(test_raw, num_classes)

    train_ds = wrap_pipeline(train_raw, shuffle=True, shuffle_size=int(train_counts.sum()))
    val_ds   = wrap_pipeline(val_raw)
    test_ds  = wrap_pipeline(test_raw)

    # --- Visuals: dataset distributions & grids & augmentations
    viz_dir = os.path.join(MODEL_DIR, "viz")
    plot_distribution(class_names, train_counts, "Train Class Distribution", viz_dir, "train_dist.png")
    plot_distribution(class_names, val_counts,   "Validation Class Distribution", viz_dir, "val_dist.png")
    plot_distribution(class_names, test_counts,  "Test Class Distribution", viz_dir, "test_dist.png")

    show_batch(train_raw, id_to_class, "Sample Training Images", save_dir=viz_dir, fname="train_grid.png")

    augmenter = get_augmenter(augment)
    if augment:
        visualize_augmentations(train_raw, augmenter, id_to_class, n=8, save_dir=viz_dir, fname="augment_preview.png")

    # --- Build & train
    model = build_model(class_names, MODEL_DIR)

    # insert augmentation into model via preprocessing (optional simple way)
    if augment:
        aug_input = tf.keras.Input(shape=IMG_SIZE+(3,))
        x = augmenter(aug_input)
        x = model(x)
        model = tf.keras.Model(aug_input, x)
        model.compile(optimizer=tf.keras.optimizers.Adam(BASE_LR),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    class_weight = {i: float(train_counts.sum()/(num_classes*max(1, train_counts[i]))) for i in range(num_classes)}
    print("Class weights:", class_weight)

    cbs = callbacks(MODEL_DIR)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs,
                        class_weight=class_weight, verbose=1)
    save_history_csv(history, MODEL_DIR)
    plot_history(history, save_dir=viz_dir)

    # Export best model
    export_model(model, MODEL_DIR)

    # --- Evaluation visuals (on test; fallback to val if empty)
    eval_ds_raw = test_raw if len(list(test_raw)) > 0 else val_raw
    eval_ds = wrap_pipeline(eval_ds_raw)
    y_true, y_pred, y_prob = collect_preds(model, eval_ds, num_classes)

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plot_cm(cm, class_names, "Confusion Matrix (Counts)", save_dir=viz_dir, fname="cm_counts.png")
    plot_cm(cm, class_names, "Confusion Matrix (Normalized)", save_dir=viz_dir, fname="cm_norm.png", normalize=True)

    # Classification report
    save_classification_report(y_true, y_pred, class_names, viz_dir)

    # ROC–AUC
    try:
        y_true_oh = tf.one_hot(y_true, depth=num_classes).numpy()
        plot_roc_all(y_true_oh, y_prob, class_names, save_dir=viz_dir)
    except Exception as e:
        print("ROC–AUC skipped:", e)

    # Prediction gallery (misclassifications in red first)
    # collect small batch of visuals
    imgs, t, p, conf = [], [], [], []
    for xb, yb in eval_ds:
        probs = model.predict(xb, verbose=0)
        yhat  = np.argmax(probs, axis=1)
        cmax  = probs[np.arange(len(yhat)), yhat]
        for i in range(len(yhat)):
            if len(imgs) >= 16: break
            imgs.append(xb[i].numpy().astype("uint8"))
            t.append(int(yb.numpy()[i])); p.append(int(yhat[i])); conf.append(float(cmax[i]))
        if len(imgs) >= 16: break
    imgs = np.array(imgs); t = np.array(t); p = np.array(p); conf = np.array(conf)
    wrong = (t != p)
    order = np.argsort(~wrong)   # misclassified first
    prediction_gallery(imgs[order], t[order], p[order], conf[order], wrong[order],
                       id_to_class, max_cols=4, title="Predictions (Green=Correct, Red=Wrong)",
                       save_dir=viz_dir, fname="prediction_gallery.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train image classifier and produce all visuals.")
    ap.add_argument("--image_dir", type=str, default=None, help="Root that has train/val/test or a single root if SINGLE_ROOT used.")
    ap.add_argument("--single_root", type=str, default=None, help="Use this if you only have one root with class folders.")
    ap.add_argument("--tfdata_path", type=str, default=None, help="(Optional) Not used; reserved for future TFRecords.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--augment", type=int, default=1, help="1 enable, 0 disable")
    args = ap.parse_args()
    main(args)
