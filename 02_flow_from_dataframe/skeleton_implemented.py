"""
==========================================================
Deep Learning Image Classification Pipeline
==========================================================
Author: Asifuzzaman Lasker
Institution: Bionary Research & AI Academy
----------------------------------------------------------
End-to-end modular deep learning pipeline for:
    • Dataset preparation and CSV generation
    • Train/Validation/Test data visualization
    • ImageDataGenerator creation
    • CNN model design, training, and evaluation
    • Performance visualization (Confusion Matrix, ROC-AUC)
    • Sample predictions
==========================================================
"""

# ==========================================================
# 01. IMPORTS
# ==========================================================
import os
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf

# ✅ Universal TensorFlow/Keras Compatibility
try:
    from keras.utils import load_img, img_to_array
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from tensorflow.keras.utils import load_img, img_to_array
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# 02. CONFIGURATION
# ==========================================================
def configure_environment():
    """Define and return pipeline configuration parameters."""
    config = {
        "main_dir": "dataset_4_100img",
        "csv_name": "image_data.csv",
        "image_size": (224, 224),
        "batch_size": 16,
        "epochs": 30,
        "val_split_ratio": 0.2,
        "test_ratio_from_temp": 0.5,
        "random_state": SEED,
        "model_save_path": "best_model.h5",
        "save_plots": True,
        "results_dir": "results"
    }
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
    return config


# ==========================================================
# 03. DATASET PREPARATION & CSV CREATION
# ==========================================================
def prepare_dataset(config):
    """
    Check if CSV exists, otherwise create it by scanning image folders.
    """
    csv_path = os.path.join(config["main_dir"], config["csv_name"])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ CSV found: {csv_path}")
    else:
        print(f"⚙️ Creating CSV from folder structure...")
        file_paths, labels = [], []
        for label in sorted(os.listdir(config["main_dir"])):
            class_dir = os.path.join(config["main_dir"], label)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_paths.append(os.path.abspath(os.path.join(class_dir, file)))
                        labels.append(label)
        df = pd.DataFrame({'filename': file_paths, 'class': labels})
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV created: {csv_path} ({len(df)} records)")

    # Ensure absolute paths
    df["filename"] = df["filename"].apply(lambda p: os.path.abspath(p))
    df = df.sample(frac=1.0, random_state=config["random_state"]).reset_index(drop=True)
    return df


# ==========================================================
# 04. TRAIN / VALIDATION / TEST SPLIT & GENERATOR CREATION
# ==========================================================
def create_generators(df, config):
    """
    Split dataset into train/val/test and create ImageDataGenerators.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=config["val_split_ratio"],
        stratify=df["class"],
        random_state=config["random_state"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config["test_ratio_from_temp"],
        stratify=temp_df["class"],
        random_state=config["random_state"]
    )

    img_h, img_w = config["image_size"]
    batch_size = config["batch_size"]

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=config["random_state"]
    )
    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filename",
        y_col="class",
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filename",
        y_col="class",
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return train_gen, val_gen, test_gen, train_df, val_df, test_df


# ==========================================================
# 05. DATA VISUALIZATION
# ==========================================================
def visualize_data_distribution(train_df, val_df, test_df):
    """Show class distribution across splits."""
    plt.figure(figsize=(14, 4))
    for i, (df, title) in enumerate(zip([train_df, val_df, test_df], ["Train", "Validation", "Test"]), 1):
        plt.subplot(1, 3, i)
        sns.countplot(data=df, x="class", order=df["class"].value_counts().index)
        plt.title(title)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_sample_images(generator, num_images=16):
    """Display random sample images from a generator."""
    x_batch, y_batch = next(generator)
    labels = np.argmax(y_batch, axis=1)
    class_map = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow(x_batch[i])
        plt.title(class_map[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ==========================================================
# 06. GENERATOR INFORMATION
# ==========================================================
def display_generator_info(train_gen, val_gen, test_gen):
    """Print details of the generators."""
    print("\n=== Generator Info ===")
    for name, g in zip(["Train", "Validation", "Test"], [train_gen, val_gen, test_gen]):
        print(f"{name} Samples: {g.samples} | Classes: {len(g.class_indices)} | Steps/Epoch: {math.ceil(g.samples/g.batch_size)}")


# ==========================================================
# 07. MODEL CREATION
# ==========================================================
def build_cnn_model(input_shape, num_classes):
    """Build and compile CNN."""
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.15),

        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),

        Conv2D(128, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


# ==========================================================
# 08. MODEL TRAINING
# ==========================================================
def train_model(model, train_gen, val_gen, config):
    """Train CNN model."""
    callbacks = [
        ModelCheckpoint(config["model_save_path"], monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1
    )
    return history


# ==========================================================
# 09. TRAINING HISTORY VISUALIZATION
# ==========================================================
def plot_training_history(history):
    """Plot accuracy and loss curves."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 10. MODEL EVALUATION DASHBOARD
# ==========================================================
def evaluate_model_dashboard(model, val_gen, class_labels):
    """Evaluate model and show Confusion Matrix, Classification Report, ROC–AUC."""
    y_prob = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = val_gen.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Classification Report
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # ROC–AUC Curves
    n_classes = len(class_labels)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(class_labels):
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC={roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC–AUC Curves")
    plt.legend()
    plt.show()


# ==========================================================
# 11. SAMPLE PREDICTIONS VISUALIZATION
# ==========================================================
def show_sample_predictions(model, val_df, class_labels, config, num_samples=9):
    """Display sample predictions with true labels."""
    sample_df = val_df.sample(num_samples, random_state=config["random_state"]).reset_index(drop=True)
    plt.figure(figsize=(num_samples*3, 3))
    for i, row in sample_df.iterrows():
        img_path, true_class = row["filename"], row["class"]
        img = load_img(img_path, target_size=config["image_size"])
        arr = img_to_array(img) / 255.0
        preds = model.predict(np.expand_dims(arr, 0), verbose=0)
        pred_idx = np.argmax(preds)
        pred_label = class_labels[pred_idx]
        confidence = np.max(preds) * 100
        plt.subplot(1, num_samples, i+1)
        plt.imshow(arr)
        plt.axis("off")
        color = "green" if pred_label == true_class else "red"
        plt.title(f"T:{true_class}\nP:{pred_label}\n{confidence:.1f}%", color=color, fontsize=9)
    plt.tight_layout()
    plt.show()


# ==========================================================
# 12. MAIN PIPELINE
# ==========================================================
def main():
    """Orchestrate the full pipeline."""
    config = configure_environment()
    df = prepare_dataset(config)
    train_gen, val_gen, test_gen, train_df, val_df, test_df = create_generators(df, config)

    visualize_data_distribution(train_df, val_df, test_df)
    visualize_sample_images(train_gen)
    display_generator_info(train_gen, val_gen, test_gen)

    input_shape = (*config["image_size"], 3)
    num_classes = len(train_gen.class_indices)
    class_labels = [None] * num_classes
    for k, v in train_gen.class_indices.items():
        class_labels[v] = k

    model = build_cnn_model(input_shape, num_classes)
    history = train_model(model, train_gen, val_gen, config)
    plot_training_history(history)

    print("\nValidation Evaluation:")
    evaluate_model_dashboard(model, val_gen, class_labels)

    print("\nSample Predictions:")
    show_sample_predictions(model, val_df, class_labels, config, num_samples=6)


# ==========================================================
# 13. SCRIPT ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    main()
