# tf_record_pipeline/predict.py
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# from .config import LABELS_JSON, IMG_SIZE
# from .visualize import prediction_gallery
# from .model_builder import build_model
# from .config import FINAL_MODEL_PATH, BEST_MODEL_PATH


from config import LABELS_JSON, IMG_SIZE, FINAL_MODEL_PATH, BEST_MODEL_PATH
from visualize import prediction_gallery

def _load_classes():
    with open(LABELS_JSON) as f:
        return json.load(f)["classes"]

def _load_model():
    if BEST_MODEL_PATH.exists():
        return tf.keras.models.load_model(str(BEST_MODEL_PATH))
    return tf.keras.models.load_model(str(FINAL_MODEL_PATH))

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    return [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]

def main():
    ap = argparse.ArgumentParser(description="Predict on a folder and show gallery")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    classes = _load_classes()
    model = _load_model()

    # build temporary items with dummy true labels (-1) to color all titles as 'unknown correctness'
    imgs = list_images(args.image_dir)
    if not imgs:
        raise SystemExit(f"No images found under: {args.image_dir}")

    # If you want correctness colors, you can pass true labels by mirroring folder names to classes.
    # Here we try to infer true labels from subfolder names if they match class names.
    items = []
    for p in imgs:
        true = -1
        # infer label from parent dir (if present in classes)
        parent = Path(p).parent.name
        if parent in classes:
            true = classes.index(parent)
        items.append((p, true if true >= 0 else 0))  # put 0 if unknown (will color red/green accordingly)

    prediction_gallery(model, items, classes, rows=args.rows, cols=args.cols)

if __name__ == "__main__":
    main()
