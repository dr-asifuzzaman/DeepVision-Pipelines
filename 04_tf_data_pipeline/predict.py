# predict.py
import argparse, os, json
import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE, MODEL_DIR, DETERMINISTIC
from data_loader import infer_structure, build_datasets
from train_utils import wrap_pipeline
from visualize import prediction_gallery

def load_label_map(model_dir):
    with open(os.path.join(model_dir, "label_map.json")) as f:
        d = json.load(f)
    id_to_class = {int(k): v for k, v in d.items()}
    class_names = [id_to_class[i] for i in sorted(id_to_class)]
    return class_names, id_to_class

def load_model(model_dir):
    sm_dir = os.path.join(model_dir, "best_savedmodel")
    try:
        return tf.keras.models.load_model(sm_dir)
    except Exception:
        return tf.saved_model.load(sm_dir)

def load_img(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)  # model expects float32 [0..255]
    arr = np.expand_dims(arr, 0)
    return img, arr

def single_image_predict(args):
    class_names, id_to_class = load_label_map(args.model_dir)
    model = load_model(args.model_dir)

    img, arr = load_img(args.image, IMG_SIZE)
    probs = model.predict(arr, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    pred_cls = id_to_class[pred_id]; conf = float(probs[pred_id])

    print(f"Prediction: {pred_cls} (confidence: {conf:.3f})")
    # Optional: small bar plot
    import matplotlib.pyplot as plt
    k = min(args.topk, len(probs))
    topk_idx = np.argsort(-probs)[:k]
    topk_labels = [id_to_class[int(i)] for i in topk_idx]
    topk_values = probs[topk_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img); axes[0].axis("off")
    axes[0].set_title(f"Pred: {pred_cls} ({conf:.1%})")
    axes[1].barh(range(k), topk_values[::-1])
    axes[1].set_yticks(range(k)); axes[1].set_yticklabels(topk_labels[::-1])
    axes[1].set_xlim(0,1); axes[1].set_xlabel("Probability")
    axes[1].set_title("Top-k")
    plt.tight_layout()
    out = os.path.join(args.model_dir, "viz_predict")
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, "single_image_prediction.png"), dpi=150)
    plt.show()

def gallery_from_test(args):
    # build eval ds
    struct = infer_structure(args.image_dir, args.single_root)
    class_names, _, _, test_raw = build_datasets(struct)
    id_to_class = {i:c for i,c in enumerate(class_names)}
    opts = tf.data.Options(); opts.experimental_deterministic = DETERMINISTIC
    test_raw = test_raw.with_options(opts)

    model = load_model(args.model_dir)
    eval_ds = wrap_pipeline(test_raw)

    # gather N = rows*cols
    want = args.rows * args.cols
    imgs, t, p, conf = [], [], [], []
    for xb, yb in eval_ds:
        probs = model.predict(xb, verbose=0)
        yhat  = np.argmax(probs, axis=1)
        cmax  = probs[np.arange(len(yhat)), yhat]
        for i in range(len(yhat)):
            if len(imgs) >= want: break
            imgs.append(xb[i].numpy().astype("uint8"))
            t.append(int(yb.numpy()[i])); p.append(int(yhat[i])); conf.append(float(cmax[i]))
        if len(imgs) >= want: break
    imgs = np.array(imgs); t = np.array(t); p = np.array(p); conf = np.array(conf)
    wrong = (t != p)
    order = np.argsort(~wrong)

    viz_dir = os.path.join(args.model_dir, "viz_predict")
    prediction_gallery(imgs[order], t[order], p[order], conf[order], wrong[order],
                       id_to_class, max_cols=args.cols, title="Prediction Gallery (Test)",
                       save_dir=viz_dir, fname="prediction_gallery.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict: gallery from test or single image.")
    ap.add_argument("--model_dir", type=str, default=MODEL_DIR)
    ap.add_argument("--image_dir", type=str, default=None, help="Required for gallery mode.")
    ap.add_argument("--single_root", type=str, default=None)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--image", type=str, default=None, help="Single image path.")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.image:
        single_image_predict(args)
    else:
        if not args.image_dir:
            raise SystemExit("--image_dir is required for gallery mode (no --image provided).")
        gallery_from_test(args)
