# main.py
import argparse
import numpy as np

from data_loader import make_dataloaders
from model_builder import build_mobilenet_v2, get_device
from train_utils import train_fit, predict_all
from visualize import (
    dataset_distribution, classwise_grid, batch_preview, plot_training_curves,
    plot_confusion_matrices, show_classification_report, plot_roc_auc, prediction_gallery
)
from config import ARTIFACTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Train + Visualize PyTorch DataLoader Pipeline")
    parser.add_argument("--image_dir", type=str, required=True, help="Root dataset folder (subdirs=classes)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    # Load data
    train_loader, val_loader, test_loader, info = make_dataloaders(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        augment=bool(args.augment),
        img_size=args.img_size,
    )
    class_names = info["class_names"]
    dataset = info["dataset"]
    targets_all = np.array([t for _, t in dataset.samples])

    # 1) Dataset distribution (All)
    dataset_distribution(targets_all, class_names, "Dataset Distribution (All)", "01_dataset_distribution.png")

    # 2) Class-wise grids (sample few images per class)
    image_paths = [p for (p, _) in dataset.samples]
    labels = [t for (_, t) in dataset.samples]
    classwise_grid(image_paths, labels, class_names, per_class=6, filename="02_classwise_grid.png")

    # 3) Augmentation preview (from train loader)
    title = "Train Batch Preview (Augmentations Applied)" if args.augment else "Train Batch Preview (No Augment)"
    batch_preview(train_loader, title=title, filename="03_augmentation_preview.png")

    # Build model & train
    device = get_device()
    model = build_mobilenet_v2(num_classes=len(class_names), pretrained=True).to(device)

    history = train_fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=1e-3,
        device=device
    )

    # 4) Curves
    plot_training_curves(history, filename="04_training_curves.png")

    # 5) Evaluation on test
    probs, preds, labels_true, imgs_tensor = predict_all(model, test_loader, device)

    # 6) Confusion matrix (counts + normalized)
    plot_confusion_matrices(labels_true, preds, class_names, filename="05_confusion_matrices.png")

    # 7) Classification report
    show_classification_report(labels_true, preds, class_names, filename="06_classification_report.txt")

    # 8) ROC-AUC
    plot_roc_auc(labels_true, probs, class_names, filename="07_roc_auc.png")

    # 9) Prediction gallery
    prediction_gallery(imgs_tensor, labels_true, preds, class_names,
                       rows=3, cols=6, filename="08_prediction_gallery.png")

    print(f"\nAll outputs saved under: {ARTIFACTS_DIR.resolve()}\n"
          "Close the current figure window to proceed to the next one.")


if __name__ == "__main__":
    main()
