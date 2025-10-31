# data_loader.py
import os
import pathlib
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, SEED, VAL_SPLIT, TEST_FRACTION_OF_VAL

def has_dir(path):
    p = pathlib.Path(path)
    return p.exists() and any(p.iterdir())

def list_classes_from_dir(directory):
    p = pathlib.Path(directory)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class subfolders found inside: {directory}")
    return classes

def infer_structure(data_root, single_root):
    if single_root and has_dir(single_root):
        return {"mode": "single", "root": single_root, "train": None, "val": None, "test": None}
    if data_root and has_dir(data_root):
        t, v, s = [os.path.join(data_root, x) for x in ["train", "val", "test"]]
        return {
            "mode": "tv",
            "root": data_root,
            "train": t if has_dir(t) else data_root,
            "val": v if has_dir(v) else None,
            "test": s if has_dir(s) else None,
        }
    raise ValueError("Please set a valid DATA_ROOT or SINGLE_ROOT.")

def build_datasets(struct):
    if struct["mode"] == "single":
        class_names = list_classes_from_dir(struct["root"])
        train = tf.keras.utils.image_dataset_from_directory(
            struct["root"], validation_split=VAL_SPLIT, subset="training",
            seed=SEED, labels="inferred", label_mode="int", class_names=class_names,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
        val_full = tf.keras.utils.image_dataset_from_directory(
            struct["root"], validation_split=VAL_SPLIT, subset="validation",
            seed=SEED, labels="inferred", label_mode="int", class_names=class_names,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
        vb = len(val_full)
        tb = max(1, int(round(vb * TEST_FRACTION_OF_VAL)))
        test = val_full.take(tb)
        val  = val_full.skip(tb)
        return class_names, train, val, test

    class_names = list_classes_from_dir(struct["train"])

    if struct["val"]:
        train = tf.keras.utils.image_dataset_from_directory(
            struct["train"], seed=SEED, labels="inferred", label_mode="int",
            class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
        val = tf.keras.utils.image_dataset_from_directory(
            struct["val"], seed=SEED, labels="inferred", label_mode="int",
            class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train = tf.keras.utils.image_dataset_from_directory(
            struct["train"], validation_split=VAL_SPLIT, subset="training",
            seed=SEED, labels="inferred", label_mode="int",
            class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
        val = tf.keras.utils.image_dataset_from_directory(
            struct["train"], validation_split=VAL_SPLIT, subset="validation",
            seed=SEED, labels="inferred", label_mode="int",
            class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

    if struct["test"]:
        test = tf.keras.utils.image_dataset_from_directory(
            struct["test"], seed=SEED, labels="inferred", label_mode="int",
            class_names=class_names, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    else:
        vb = len(val)
        tb = max(1, int(round(vb * TEST_FRACTION_OF_VAL)))
        test = val.take(tb)
        val  = val.skip(tb)

    return class_names, train, val, test
