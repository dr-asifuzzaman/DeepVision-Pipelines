# TensorFlow tf.data Pipeline

Modern, efficient, and production-ready. This is how TensorFlow wants you to load data, and for good reason.

## Why tf.data Exists

TensorFlow created `tf.data` to solve a fundamental problem: your GPU shouldn't wait for data. Traditional approaches load data on the CPU, preprocess it, then send batches to the GPU. During this time, your expensive GPU sits idle.

`tf.data` pipelines overlap data loading, preprocessing, and training. While your GPU trains on batch N, the CPU prepares batch N+1. The result? Up to 3x faster training on the same hardware.

## When To Use This

Use `tf.data` when:
- You're serious about training performance
- Working with medium to large datasets
- Building production ML systems
- Need reproducible data pipelines
- Want to leverage multiple CPU cores efficiently
- Training on GPU/TPU and want to maximize utilization

Stick with simpler approaches if:
- You're just learning and want something straightforward
- Dataset is tiny (< 1000 images) and speed doesn't matter
- You're debugging and prefer simple, synchronous loading
- Not using TensorFlow at all

## Core Concepts

**Datasets are lazy**: `tf.data.Dataset` doesn't load anything until you iterate. It represents a pipeline of operations that will execute when needed.

**Chaining operations**: Build pipelines by chaining transformations. Each step transforms the dataset in some way.

**Automatic optimization**: TensorFlow analyzes your pipeline and applies optimizations automatically. Things like prefetching and parallel processing happen behind the scenes.

**Deterministic by default**: Set a seed and get the same data order every time. Crucial for reproducibility.


# 🧠 TensorFlow `tf.data` Image Classification Pipeline

Modern, efficient, and production-ready — this is how TensorFlow *wants* you to load data, and for good reason.  
This repository implements a **modular image classification system** using `tf.data`, `MobileNetV2`, and full visualization support — from dataset inspection to ROC–AUC analysis.

---

```
04_tf_data_pipeline/
├── config.py                # Global constants & settings
├── data_loader.py           # tf.data dataset construction
├── visualize.py             # Dataset previews & plots
├── model_builder.py         # MobileNetV2 (base) + classifier head
├── train_utils.py           # Class weights, callbacks, and model export
├── main.py                  # CLI for training + visualization
├── evaluate.py              # Evaluate saved model
├── predict.py               # Predict on test set or single image
└── model_artifacts/         # Saved model & plots

```

## How To Run the Pipeline

```
python main.py \
  --image_dir /path/to/dataset \
  --epochs 10 \
  --batch_size 32 \
  --augment 1
```

####  What happens automatically:

- Loads dataset via tf.data

- Splits into train / val / test

- Builds efficient pipeline (cache → shuffle → batch → prefetch)

- Applies augmentations (if --augment 1)

- Starts model training

- Generates and saves:

    - Dataset distribution graphs

    - Class-wise grids

    - Augmentation previews

    - Training & validation curves

    - Confusion matrix (counts + normalized)

    - Classification report

    - ROC–AUC plots (per-class + micro/macro)

    - Prediction gallery (correct = green, wrong = red)

#### All outputs go into:
``` model_artifacts/viz/``

#### Evaluate after training
``` python evaluate.py --image_dir /path/to/dataset --rows 2 --cols 5```

#### Predict on test or single image
``` python predict.py --image_dir /path/to/dataset --rows 2 --cols 5 ```
