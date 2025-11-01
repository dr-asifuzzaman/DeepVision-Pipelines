# tf_record_pipeline/data_loader.py
import math, json, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import tensorflow as tf

from config import (
    IMG_SIZE, CHANNELS, SEED, SHUFFLE_BUFFER,
    USE_GZIP_TFRECORDS, RECORDS_PER_SHARD,
    TFRECORD_DIR, LABELS_JSON
)

AUTOTUNE = tf.data.AUTOTUNE

# ---------- Files & classes ----------
def list_classes(root_dir: str) -> List[str]:
    classes = [p.name for p in sorted(Path(root_dir).glob("*")) if p.is_dir()]
    if not classes:
        raise ValueError(f"No class subdirectories found in: {root_dir}")
    return classes

def list_images_with_labels(root_dir: str, classes: List[str]) -> List[Tuple[str, int]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    items = []
    for idx, cname in enumerate(classes):
        for p in Path(root_dir, cname).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((str(p), idx))
    return items

def split_items(all_items, val_split: float, test_split: float):
    random.shuffle(all_items)
    n_total = len(all_items)
    n_test = int(round(test_split * n_total))
    n_val  = int(round(val_split  * n_total))
    n_train = n_total - n_val - n_test
    train_items = all_items[:n_train]
    val_items   = all_items[n_train:n_train+n_val]
    test_items  = all_items[n_train+n_val:]
    return train_items, val_items, test_items

# ---------- TFRecord helpers ----------
def _bytes_feature(v: bytes): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _int64_feature(v: int):   return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

def _encode_image(path: str, target_size, channels: int) -> bytes:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size, antialias=True)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img_u8 = tf.image.convert_image_dtype(img, tf.uint8)
    return tf.io.encode_jpeg(img_u8, quality=95).numpy()

def _serialize_example(img_bytes: bytes, label: int, h: int, w: int, c: int) -> bytes:
    ex = tf.train.Example(features=tf.train.Features(feature={
        "image": _bytes_feature(img_bytes),
        "label": _int64_feature(label),
        "height": _int64_feature(h),
        "width":  _int64_feature(w),
        "channels": _int64_feature(c),
    }))
    return ex.SerializeToString()

def write_split_tfrecord(split_name: str, items: List[Tuple[str, int]]):
    if not items: return
    TFRECORD_DIR.mkdir(parents=True, exist_ok=True)
    compression = "GZIP" if USE_GZIP_TFRECORDS else None
    options = tf.io.TFRecordOptions(compression_type=compression) if compression else None

    num_shards = max(1, math.ceil(len(items) / RECORDS_PER_SHARD))
    shard_sizes = [len(items) // num_shards] * num_shards
    for i in range(len(items) % num_shards):
        shard_sizes[i] += 1

    start = 0
    for shard_id, shard_size in enumerate(shard_sizes):
        end = start + shard_size
        shard_items = items[start:end]
        start = end
        shard_path = TFRECORD_DIR / f"{split_name}-{shard_id:03d}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_path), options=options) as w:
            for p, lab in shard_items:
                try:
                    img_bytes = _encode_image(p, IMG_SIZE, CHANNELS)
                    w.write(_serialize_example(img_bytes, lab, IMG_SIZE[0], IMG_SIZE[1], CHANNELS))
                except Exception as e:
                    print(f"[WARN] failed encoding {p}: {e}")
        print(f"Wrote {split_name} -> {shard_path}")

def ensure_tfrecords(image_dir: str, classes: List[str], val_split: float, test_split: float):
    # If tfrecords already exist, skip creation
    existing = list(TFRECORD_DIR.glob("*.tfrecord"))
    if existing:
        print(f"Found {len(existing)} TFRecord shards in {TFRECORD_DIR}, reusing.")
        return None, None, None  # unknown counts

    items = list_images_with_labels(image_dir, classes)
    train_items, val_items, test_items = split_items(items, val_split, test_split)

    # save labels map
    LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_JSON, "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    write_split_tfrecord("train", train_items)
    write_split_tfrecord("val",   val_items)
    write_split_tfrecord("test",  test_items)
    return train_items, val_items, test_items

# ---------- Dataset readers ----------
_feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width":  tf.io.FixedLenFeature([], tf.int64),
    "channels": tf.io.FixedLenFeature([], tf.int64),
}

def _parse_example(rec):
    ex = tf.io.parse_single_example(rec, _feature_desc)
    img = tf.io.decode_jpeg(ex["image"], channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    lab = tf.cast(ex["label"], tf.int32)
    return img, lab

def _augment(img, lab):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, lab

def make_dataset(split: str, batch_size: int, training: bool):
    compression_type = "GZIP" if USE_GZIP_TFRECORDS else None
    files = tf.io.gfile.glob(str(TFRECORD_DIR / f"{split}-*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for split '{split}' at {TFRECORD_DIR}")
    ds = tf.data.TFRecordDataset(files, compression_type=compression_type, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_example, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
