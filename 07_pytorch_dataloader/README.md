# PyTorch DataLoader Pipeline

PyTorch's approach to data loading. If you're building with PyTorch instead of TensorFlow, this is your home.

## PyTorch vs TensorFlow Philosophy

PyTorch and TensorFlow handle data differently:

**PyTorch**: Pythonic and explicit. You write Python classes that define how data loads. Debugging feels natural because it's just Python code.

**TensorFlow**: Graph-based and optimized. You build pipelines that TensorFlow optimizes automatically. More magic, potentially faster.

Neither is better - they're different tools for different preferences. This pipeline shows the PyTorch way.

## When To Use PyTorch DataLoader

Use PyTorch DataLoader when:
- Your model is built in PyTorch
- You prefer explicit, Pythonic code
- Working in research where flexibility matters
- Need easy debugging with standard Python tools
- Team is familiar with PyTorch ecosystem

Stick with TensorFlow if:
- Using TensorFlow/Keras for modeling
- Need maximum performance optimization
- Working with TPUs (PyTorch support is limited)
- Deploying to TensorFlow Serving

## Core Components

PyTorch data loading has two main parts:

**Dataset**: Defines how to access individual samples
- Implements `__len__()` and `__getitem__()`
- Just returns one sample at a time
- No batching, no shuffling

**DataLoader**: Handles batching, shuffling, parallel loading
- Wraps your Dataset
- Creates batches
- Shuffles data
- Loads in parallel with multiple workers


## 🚀 Overview

- End-to-end image classification pipeline in PyTorch with a fully modular design and CLI:

- Stratified train/val/test splits

- Augmentations (toggle on/off)

- MobileNetV2 backbone (easily swappable)

- Blocking, one-by-one pop-up plots (close a window to see the next)

- All figures auto-saved under model_artifacts/plots/

- Evaluation and Prediction as separate commands

## 📁 Project Structure

```
pytorch_dataloader/
├── config.py               # Global constants & folders
├── data_loader.py          # ImageFolder dataset + stratified splits + loaders
├── visualize.py            # All plots & previews (blocking windows + save)
├── model_builder.py        # MobileNetV2 (pretrained) + device helpers
├── train_utils.py          # Train loop, early stop, prediction helpers
├── main.py                 # Train + visualize (sequential pop-ups)
├── evaluate.py             # Evaluate saved best model + plots
├── predict.py              # Predict on test set or a single image
└── model_artifacts/        # Auto-created: checkpoints, plots, metadata
    ├── checkpoints/
    └── plots/
```

Your dataset must be organized like:

```
/path/to/dataset
├── class_A
├── class_B
└── class_C
```

## Quick Start

### 1) Train + all sequential visualizations
This command trains the model and shows pop-up windows in order.
Close each window to proceed to the next. All plots are also saved.

```
python main.py \
  --image_dir /path/to/dataset \
  --epochs 10 \
  --batch_size 32 \
  --augment 1 \
  --img_size 224
```

What you’ll see (and what gets saved under model_artifacts/plots/):

1. Dataset distribution (bar chart)
2. Class-wise grids (samples per class)
3. Augmentation preview (train batch)
4. Training & validation curves (loss/accuracy)
5. Confusion matrices (counts + normalized)
6. Classification report (also saved as .txt)
7. ROC–AUC (per-class + micro/macro)
8. Prediction gallery (✅ green correct / ❌ red wrong)


Artifacts:

- Best model: ```model_artifacts/checkpoints/best_model.pt```
- Training history: ```model_artifacts/train_history.json```
- Class map: ```model_artifacts/class_mapping.json``
- Split indices: ```model_artifacts/split_indices.json```


### 2) Evaluate (on the saved best model)

```
python evaluate.py \
  --image_dir /path/to/dataset \
  --rows 2 \
  --cols 5 \
  --batch_size 32 \
  --img_size 224 

```

Shows/saves:

- Confusion matrices
- Classification report
- ROC–AUC
- Prediction gallery (rows × cols images)


### 3) Predict
A. Use samples from the test split (gallery):

```
python predict.py \
  --image_dir /path/to/dataset \
  --rows 2 \
  --cols 5 \
  --batch_size 32 \
  --img_size 224
```

B. Predict a single image:
```
python predict.py \
  --image_dir /path/to/dataset \
  --image_path /path/to/one.jpg \
  --img_size 224
```

### 🧷 Reproducibility

Stratified splits are saved to model_artifacts/split_indices.json.

Class map is saved to model_artifacts/class_mapping.json.

Training history saved to model_artifacts/train_history.json.

## ❓ FAQ

Q: My dataset is imbalanced; can I use class weights?
A: Yes—add weight= to nn.CrossEntropyLoss() in train_utils.py using per-class weights from counts.

Q: How do I change augmentations?
A: Edit train_transform() in data_loader.py.

Q: Where are figures saved?
A: model_artifacts/plots/. Filenames are ordered (01_..., 02_..., etc.).

Q: How do I use GPU?
A: If torch.cuda.is_available() is true, the code will use CUDA automatically.