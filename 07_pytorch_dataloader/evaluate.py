# evaluate.py
import argparse
import json
import numpy as np
import torch

from data_loader import make_dataloaders
from model_builder import build_mobilenet_v2, get_device
from train_utils import predict_all
from visualize import (
    plot_confusion_matrices, show_classification_report, plot_roc_auc, prediction_gallery
)
from config import BEST_CKPT


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved best model")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    # Only need loaders to get class names and test set
    _, _, test_loader, info = make_dataloaders(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        augment=False,
        img_size=args.img_size
    )
    class_names = info["class_names"]

    # Build and load model
    device = get_device()
    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=False).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Predict
    probs, preds, labels_true, imgs_tensor = predict_all(model, test_loader, device)

    # Visuals
    plot_confusion_matrices(labels_true, preds, class_names, filename="E_confusion_matrices.png")
    show_classification_report(labels_true, preds, class_names, filename="E_classification_report.txt")
    plot_roc_auc(labels_true, probs, class_names, filename="E_roc_auc.png")
    prediction_gallery(imgs_tensor, labels_true, preds, class_names,
                       rows=args.rows, cols=args.cols, filename="E_prediction_gallery.png")


if __name__ == "__main__":
    main()
