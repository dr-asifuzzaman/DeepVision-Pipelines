# tf_record_pipeline/main.py
import argparse, json, os
import numpy as np
import tensorflow as tf

from config import (
    IMG_SIZE, CHANNELS, VAL_SPLIT, TEST_SPLIT, SEED, USE_MIXED_PRECISION
)


from data_loader import list_classes, ensure_tfrecords, make_dataset
from visualize import plot_split_distribution, classwise_grid, show_augmentation_preview
from model_builder import build_model, fine_tune
from train_utils import save_labels, compute_weights_from_items, callbacks, plot_history, export_final

def main():
    p = argparse.ArgumentParser(description="Train classifier with TFRecords + tf.data")
    p.add_argument("--image_dir", required=True, help="Dataset root (class subfolders)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--augment", type=int, default=1, help="1 to use data augment in pipeline")
    args = p.parse_args()

    if USE_MIXED_PRECISION:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")

    # Classes & TFRecords
    classes = list_classes(args.image_dir)
    save_labels(classes)

    train_items, val_items, test_items = ensure_tfrecords(
        args.image_dir, classes, VAL_SPLIT, TEST_SPLIT
    )

    # Visuals in requested order (each blocks until closed)
    plot_split_distribution(train_items, val_items, test_items, classes)
    classwise_grid(train_items or [], classes, k_per_class=5)
    show_augmentation_preview(train_items or [])

    # Datasets
    train_ds = make_dataset("train", args.batch_size, training=bool(args.augment))
    val_ds   = make_dataset("val",   args.batch_size, training=False)

    # Model
    model = build_model(num_classes=len(classes))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class_weights = compute_weights_from_items(train_items, len(classes)) if train_items else None

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks()
    )

    #optional fine-tuning
    # model = fine_tune(model, lr=1e-4)
    # history_ft = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=max(3, args.epochs // 2),
    #     callbacks=callbacks()
    # )

    # Curves
    plot_history(history)

    # Save final
    export_final(model)

if __name__ == "__main__":
    main()
