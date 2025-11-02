# data_loader.py
import os, glob, random, itertools
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import Config

AUTOTUNE = tf.data.AUTOTUNE

def _list_images_by_class(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    filepaths, labels = [], []
    for idx, c in enumerate(classes):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            for fp in glob.glob(os.path.join(root_dir, c, ext)):
                filepaths.append(fp)
                labels.append(idx)
    return filepaths, labels, classes

def _has_split(image_dir: str) -> bool:
    return all(os.path.isdir(os.path.join(image_dir, split)) for split in ("train","val","test"))

def _make_splits_from_single_dir(image_dir: str, cfg: Config):
    filepaths, labels, classes = _list_images_by_class(image_dir)
    if len(filepaths) == 0:
        raise ValueError(f"No images found in {image_dir}. Expect subfolders per class.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        filepaths, labels, test_size=cfg.test_split, stratify=labels, random_state=cfg.seed
    )
    rel_val = cfg.val_split / (1.0 - cfg.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=cfg.seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes

def _read_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size, antialias=True)
    return img

def _build_dataset(
    filepaths: List[str],
    labels: List[int],
    img_size: Tuple[int,int],
    batch_size: int,
    shuffle: bool,
    augment_layer: tf.keras.Sequential = None,
    cache: bool = True,
    prefetch: int = 2,
):
    paths = tf.constant(filepaths)
    labs = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labs))

    def _load(path, lab):
        img = _read_image(path, img_size)
        return img, tf.one_hot(lab, depth=tf.reduce_max(labs)+1)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 10000), seed=1337, reshuffle_each_iteration=True)
    if cache:
        ds = ds.cache()
    if augment_layer is not None:
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(prefetch)
    return ds

def get_augmentation(cfg: Config):
    # Light, fast, GPU-friendly augmentation
    return tf.keras.Sequential([
        # tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.05),
    ], name="augment")

def load_datasets(cfg: Config):
    rng = np.random.RandomState(cfg.seed)
    if _has_split(cfg.image_dir):
        train_dir = os.path.join(cfg.image_dir, "train")
        val_dir   = os.path.join(cfg.image_dir, "val")
        test_dir  = os.path.join(cfg.image_dir, "test")
        X_train, y_train, classes = _list_images_by_class(train_dir)[0], _list_images_by_class(train_dir)[1], _list_images_by_class(train_dir)[2]
        X_val,   y_val,   _       = _list_images_by_class(val_dir)
        X_test,  y_test,  _       = _list_images_by_class(test_dir)
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = _make_splits_from_single_dir(cfg.image_dir, cfg)

    augment_layer = get_augmentation(cfg) if cfg.augment else None

    train_ds = _build_dataset(X_train, y_train, cfg.img_size, cfg.batch_size, shuffle=True,
                              augment_layer=augment_layer, cache=cfg.cache_dataset, prefetch=cfg.prefetch_batches)
    val_ds   = _build_dataset(X_val,   y_val,   cfg.img_size, cfg.batch_size, shuffle=False,
                              augment_layer=None, cache=cfg.cache_dataset, prefetch=cfg.prefetch_batches)
    test_ds  = _build_dataset(X_test,  y_test,  cfg.img_size, cfg.batch_size, shuffle=False,
                              augment_layer=None, cache=cfg.cache_dataset, prefetch=cfg.prefetch_batches)

    dist = {
        "train": np.bincount(y_train, minlength=len(classes)),
        "val":   np.bincount(y_val,   minlength=len(classes)),
        "test":  np.bincount(y_test,  minlength=len(classes)),
    }
    index = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    return train_ds, val_ds, test_ds, classes, dist, index
