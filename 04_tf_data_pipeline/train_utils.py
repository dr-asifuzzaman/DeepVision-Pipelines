# train_utils.py
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from config import MODEL_DIR, CACHE_IN_MEMORY, SEED

def count_by_class(ds, num_classes):
    counts = np.zeros(num_classes, dtype=int)
    for _, y in ds.unbatch():
        counts[int(y.numpy())] += 1
    return counts

def wrap_pipeline(ds, shuffle=False, shuffle_size=1000, seed=SEED):
    if CACHE_IN_MEMORY: ds = ds.cache()
    if shuffle: ds = ds.shuffle(buffer_size=min(shuffle_size, 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def callbacks(model_dir=MODEL_DIR):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "best_ckpt.weights"),
            monitor="val_accuracy", save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
    ]

def collect_preds(model, ds, num_classes):
    y_true, y_prob = [], []
    for xb, yb in ds:
        probs = model.predict(xb, verbose=0)
        y_prob.append(probs)
        y_true.append(yb.numpy())
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob

def export_model(model, model_dir=MODEL_DIR):
    import shutil
    ckpt_weights = os.path.join(model_dir, "best_ckpt.weights")
    best_savedmodel_dir = os.path.join(model_dir, "best_savedmodel")
    model.load_weights(ckpt_weights)

    if os.path.isdir(best_savedmodel_dir):
        shutil.rmtree(best_savedmodel_dir)
    try:
        model.export(best_savedmodel_dir)
        print("Exported model (SavedModel) to:", best_savedmodel_dir)
    except AttributeError:
        tf.saved_model.save(model, best_savedmodel_dir)
        print("Saved model (SavedModel) to:", best_savedmodel_dir)

    best_model_path = os.path.join(model_dir, "best_model.keras")
    try:
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        model.save(best_model_path)
        print("Also saved native Keras model to:", best_model_path)
    except Exception as e:
        print("Native .keras save skipped due to:", repr(e))

def save_history_csv(history, model_dir=MODEL_DIR):
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(model_dir, "history.csv"), index=False)
    print("Saved:", os.path.join(model_dir, "history.csv"))
