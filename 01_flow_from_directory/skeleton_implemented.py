
"""
Modular Flow From Directory Dataset Pipeline for
Training and Evaluating CNN Models with Tensorflow.
Author: Asifuzzaman Lasker
"""

import os
import math
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -------------------------
# 1. CONFIGURATION
# -------------------------
def configure_environment() -> Dict:
    """Return dictionary of configuration parameters (editable)."""
    cfg = {
        "DATASET_DIR": "dataset_4_100img",
        "IMG_HEIGHT": 224,
        "IMG_WIDTH": 224,
        "BATCH_SIZE": 32,
        "EPOCHS": 10,
        "MODEL_SAVE_PATH": "best_model.h5",
        "RANDOM_SEED": 42,
    }
    return cfg

# -------------------------
# 2. DIRECTORY HANDLING
# -------------------------
def find_dataset_folders(base_path: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Try to find training and validation folders with some common synonyms.
    Returns (train_dir, val_dir, use_split_flag).
      - If both found: use_split_flag = False
      - If not found: use_split_flag = True (use validation_split)
    """
    train_synonyms = ['train', 'training', 'Train', 'Training']
    val_synonyms = ['val', 'validation', 'valid', 'Validation', 'Valid']

    train_dir = None
    val_dir = None

    for name in train_synonyms:
        p = os.path.join(base_path, name)
        if os.path.isdir(p):
            train_dir = p
            break

    for name in val_synonyms:
        p = os.path.join(base_path, name)
        if os.path.isdir(p):
            val_dir = p
            break

    use_split = not (train_dir and val_dir)
    return train_dir, val_dir, use_split

# -------------------------
# 3. DATA GENERATORS
# -------------------------
def create_data_generators(
    dataset_dir: str,
    train_dir: Optional[str],
    val_dir: Optional[str],
    use_split: bool,
    img_size: Tuple[int, int],
    batch_size: int,
    seed: int = 42
):
    """
    Prepare and return (train_gen, val_gen).
    If use_split is True, uses validation_split on the single dataset_dir.
    """
    if use_split:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )

        train_gen = datagen.flow_from_directory(
            dataset_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=seed
        )
        val_gen = datagen.flow_from_directory(
            dataset_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=seed
        )
    else:
        datagen_train = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        datagen_val = ImageDataGenerator(rescale=1./255)

        train_gen = datagen_train.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=seed
        )
        val_gen = datagen_val.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=seed
        )

    return train_gen, val_gen

# -------------------------
# 4. VISUALIZATION HELPERS
# -------------------------
def visualize_sample_images(generator, num_images: int = 24):
    """Display a grid of sample images from a generator (uses next())."""
    images, labels = next(generator)
    num_images = min(num_images, len(images))
    ncols = int(min(6, num_images))
    nrows = math.ceil(num_images / ncols)
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for i in range(num_images):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {np.argmax(labels[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training & validation accuracy and loss from history."""
    if history is None:
        return
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# -------------------------
# 5. MODEL BUILDING
# -------------------------
def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """
    Create and compile a simple CNN model.
    Swap or extend this function to try other architectures.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# 6. TRAINING
# -------------------------
def train_model(model, train_gen, val_gen, epochs: int, save_path: str):
    """Train the model with callbacks and return the history."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

# -------------------------
# 7. EVALUATION
# -------------------------
def evaluate_model(model, val_gen, class_names: List[str]):
    """
    Compute predictions, print classification report and plot confusion matrix.
    Note: val_gen.shuffle must be False.
    """
    # Predict probabilities
    steps = math.ceil(val_gen.samples / val_gen.batch_size)
    y_pred_prob = model.predict(val_gen, steps=steps, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = val_gen.classes  # requires shuffle=False for correct ordering

    # Classification report
    print("\nClassification Report\n" + "="*20)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return y_true, y_pred, y_pred_prob

# -------------------------
# 8. ROC-AUC (multi-class)
# -------------------------
def plot_multiclass_roc(y_true, y_score, class_names: List[str]):
    """
    Plot ROC curves for multi-class classification.
    y_true: integer labels (shape [n_samples])
    y_score: predicted probabilities (shape [n_samples, n_classes])
    """
    n_classes = len(class_names)
    if n_classes == 2:
        # binary case
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Binary)")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        return

    # multi-class: binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # Plot
    plt.figure(figsize=(8, 8))
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{name} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr_micro, tpr_micro, linestyle=':', linewidth=3, label=f'Micro-average (AUC = {roc_auc_micro:.2f})')
    plt.plot(all_fpr, mean_tpr, linestyle=':', linewidth=3, label=f'Macro-average (AUC = {roc_auc_macro:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC")
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------
# 9. SAMPLE PREDICTIONS
# -------------------------
def show_sample_predictions(model, generator, class_names: List[str], num_predictions: int = 5):
    """Visualize model predictions for a few images from the generator."""
    images, labels = next(generator)
    preds = model.predict(images[:num_predictions])
    plt.figure(figsize=(num_predictions * 3, 3))
    for i in range(num_predictions):
        plt.subplot(1, num_predictions, i + 1)
        plt.imshow(images[i])
        true_idx = np.argmax(labels[i])
        pred_idx = np.argmax(preds[i])
        conf = preds[i][pred_idx] * 100
        plt.title(f"T:{class_names[true_idx]}\nP:{class_names[pred_idx]} ({conf:.1f}%)")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# -------------------------
# 10. MAIN PIPELINE
# -------------------------
def main():
    cfg = configure_environment()

    # find folders
    train_dir, val_dir, use_split = find_dataset_folders(cfg["DATASET_DIR"])
    print("Dataset folders ->", "Train:", train_dir, "Val:", val_dir, "Use split:", use_split)

    # create generators
    img_size = (cfg["IMG_HEIGHT"], cfg["IMG_WIDTH"])
    train_gen, val_gen = create_data_generators(
        cfg["DATASET_DIR"], train_dir, val_dir, use_split,
        img_size, cfg["BATCH_SIZE"], seed=cfg["RANDOM_SEED"]
    )

    # summary info
    class_names = list(train_gen.class_indices.keys())
    print(f"\nDataset summary: train_samples={train_gen.samples}, val_samples={val_gen.samples}, classes={class_names}")

    # visualize a few samples
    print("\nShowing sample training images...")
    visualize_sample_images(train_gen, num_images=12)

    # build model
    model = build_cnn_model((cfg["IMG_HEIGHT"], cfg["IMG_WIDTH"], 3), train_gen.num_classes)
    model.summary()

    # train
    print("\nStarting training...")
    history = train_model(model, train_gen, val_gen, cfg["EPOCHS"], cfg["MODEL_SAVE_PATH"])

    # training plots
    plot_training_history(history)

    # load best model
    if os.path.exists(cfg["MODEL_SAVE_PATH"]):
        model = load_model(cfg["MODEL_SAVE_PATH"])
        print(f"\nLoaded best model from {cfg['MODEL_SAVE_PATH']}")

    # evaluation
    y_true, y_pred, y_pred_prob = evaluate_model(model, val_gen, class_names)

    # ROC-AUC
    plot_multiclass_roc(y_true, y_pred_prob, class_names)

    # sample predictions
    print("\nSample predictions on validation images:")
    show_sample_predictions(model, val_gen, class_names, num_predictions=5)

if __name__ == "__main__":
    main()
