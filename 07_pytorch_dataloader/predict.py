# predict.py
import argparse
import numpy as np
import torch
import PIL.Image as Image
from pathlib import Path

from data_loader import make_dataloaders, base_transform
from model_builder import build_mobilenet_v2, get_device
from train_utils import predict_all
from visualize import prediction_gallery
from config import BEST_CKPT


def main():
    parser = argparse.ArgumentParser(description="Predict on test set or a single image")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None, help="Optional single image path")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    # Load class info via loaders
    _, _, test_loader, info = make_dataloaders(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        augment=False,
        img_size=args.img_size
    )
    class_names = info["class_names"]

    # Build + load model
    device = get_device()
    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=False).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.image_path:
        # Single image prediction
        tfm = base_transform(args.img_size)
        img = Image.open(args.image_path).convert("RGB")
        tensor = tfm(img).unsqueeze(0)  # [1, C, H, W]
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        print(f"Predicted: {class_names[pred_idx]} (index {pred_idx})")
        for i, cname in enumerate(class_names):
            print(f"{cname}: {probs[i]:.4f}")
    else:
        # Sample from test set to build a gallery
        probs, preds, labels_true, imgs_tensor = predict_all(model, test_loader, device)
        prediction_gallery(imgs_tensor, labels_true, preds, class_names,
                           rows=args.rows, cols=args.cols, filename="P_prediction_gallery.png")


if __name__ == "__main__":
    main()
