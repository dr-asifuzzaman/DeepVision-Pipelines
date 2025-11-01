# TFRecord Pipeline

TensorFlow's native format for large-scale machine learning. When you have millions of images, this is your path forward.

## What Are TFRecords?

TFRecords are binary files that store data in TensorFlow's native format. Think of them as highly optimized containers specifically designed for machine learning workloads.

Unlike directories of images or CSV files, TFRecords:
- Read sequentially, perfect for streaming large datasets
- Store data in a format GPUs can consume directly
- Support efficient sharding across multiple files
- Enable distributed training across many machines
- Compress data effectively while maintaining fast access

## When To Use TFRecords

You should use TFRecords when:
- Working with 100k+ images
- Dataset doesn't fit in RAM
- Training across multiple GPUs or TPUs
- Need maximum I/O performance
- Building production ML pipelines
- Dataset is stable (not changing frequently)

Don't use TFRecords if:
- Dataset is small (< 10k images)
- Still experimenting with data preprocessing
- Need to frequently add/remove samples
- Simplicity matters more than raw performance

The overhead of creating TFRecords only makes sense for large, stable datasets where the creation cost is amortized over many training runs.


# 🧠 TFRecord Image Classification Pipeline (TensorFlow + tf.data)

A **modular, end-to-end image classification pipeline** using **TensorFlow**, **tf.data**, and **TFRecord** format — designed for scalability, reproducibility, and rich visual insights.

This project automates **data loading, TFRecord creation, augmentation, training, evaluation, and visualization**, producing professional plots for model diagnostics.

---

## 📂 Folder Structure
```
tfRecord/
├── config.py # Global constants & settings
├── data_loader.py # TFRecord creation + tf.data pipelines
├── visualize.py # Dataset previews & plots (auto-save + GUI show)
├── model_builder.py # Base model (MobileNetV2) + classifier head
├── train_utils.py # Class weights, callbacks, training curves, exports
├── main.py # CLI for training + data visualization
├── evaluate.py # Evaluate model, confusion, ROC, report, gallery
├── predict.py # Predict & visualize results
└── model_artifacts/
    ├── plots/ # All generated figures (auto-saved)
    ├── model_best.keras # Best model checkpoint
    ├── model_final.keras # Final model
    ├── tfrecords/ # TFRecord shards (train/val/test)
    ├── labels.json # Saved class-to-index map
    └── train_log.csv # Training history log
```

## ⚙️ Installation
```
# Create environment
conda create -n tfrecord python=3.10 -y
conda activate tfrecord

# Install dependencies
pip install tensorflow matplotlib scikit-learn pillow numpy
```

### 🚀 Usage

#### 1. Training + Visualizations

```python main.py \
  --image_dir /path/to/dataset \
  --epochs 10 \
  --batch_size 32 \
  --augment 1
```

#### 2. Evaluation (with all performance plots)
``` python evaluate.py \
  --image_dir /path/to/dataset \
  --rows 2 --cols 5 ```