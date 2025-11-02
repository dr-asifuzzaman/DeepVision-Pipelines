# Custom Generator Pipeline

When nothing else fits your needs, build it yourself. Complete control over data loading, preprocessing, and batching.

## What Are Custom Generators?

A custom generator is a Python function or class that yields batches of data on demand. Instead of using pre-built tools like ImageDataGenerator or tf.data, you write the loading logic yourself.

This means you control:
- Exactly how files are read from disk
- Preprocessing steps and their order
- Augmentation techniques
- Batch composition
- Memory management
- Everything else

## When To Use Custom Generators

Build a custom generator when:
- Your data format is unusual (medical imaging, satellite data, etc.)
- You need preprocessing steps not available in standard tools
- Working with multiple data sources simultaneously
- Implementing research papers with specific data requirements
- Legacy code requires a specific interface
- Need to integrate external libraries (OpenCV, PIL, custom C++ code)

Don't build one if:
- Standard tools work fine for your use case
- You're just starting out and want simplicity
- Team needs maintainable, documented code
- Performance is critical (tf.data is usually faster)

The power of custom generators is flexibility. The cost is maintenance and potential performance issues if not implemented carefully.


## ⚙️ Key Features

The **Custom Generator Pipeline** offers a modern, modular, and visually interactive deep learning workflow.  
Each stage is executed sequentially — allowing you to close one visualization window before the next opens — and every output is saved under `results/`.

| 🔹 Stage | 🧭 Description |
|:---------|:---------------|
| 🗂 **Dataset Distribution Graphs** | Visualizes the number of images per class across **train**, **validation**, and **test** splits. Helps identify class imbalance before training. |
| 🧩 **Class-wise Image Grids** | Displays representative image grids from each class — great for quick dataset inspection and visual debugging. |
| 🔄 **Augmentation Previews** | Demonstrates how random augmentations (flip, rotation, zoom, contrast) transform your input samples in real-time. |
| 📈 **Training & Validation Curves** | Shows epoch-wise **loss** and **accuracy** curves for both training and validation sets to track learning behavior and overfitting. |
| 🔢 **Confusion Matrix (Counts + Normalized)** | Provides **blush-themed heatmaps** showing per-class prediction strengths and weaknesses — normalized and raw count versions. |
| 📊 **Classification Report** | Displays per-class **precision**, **recall**, and **F1-score**, rendered as a formatted text plot for easy readability. |
| 🩺 **ROC–AUC Curves (Per-class + Micro/Macro)** | Plots ROC–AUC curves for each class, along with micro and macro averages, offering a detailed performance breakdown. |
| 🖼 **Prediction Gallery (Confidence-Aware)** | Presents a visual grid of sample predictions:<br>✅ **Correct predictions** in green with confidence %<br>❌ **Misclassifications** in red with confidence %. |
| 💾 **Automatic Result Saving** | Every figure (plots, matrices, galleries) is automatically saved under `results/<category>/<timestamp>/` and the model under `model_artifacts/`. |
| 🔍 **Sequential Visualization Flow** | Each figure appears interactively — close one to open the next — ensuring a clean and organized review process. |

> 💡 **Tip:** This pipeline is designed for both research and production workflows.  
> It can easily integrate into **medical imaging**, **biological analysis**, or any **custom data domain** requiring full control and interpretability.


Each visualization is saved under:
```results/<category>/<timestamp>/```

and the best model is stored in:
```model_artifacts/best_model.keras```


## 🧩 Project Structure

```custom_generator/
├── config.py              # Global configuration
├── data_loader.py         # Custom generator (tf.data)
├── visualize.py           # Visualization utilities
├── model_builder.py       # MobileNetV2 + classifier
├── train_utils.py         # Callbacks, class weights, exports
├── main.py                # Train + visualize pipeline
├── evaluate.py            # Evaluate trained model
├── predict.py             # Predict with visualization
├── model_artifacts/       # Saved models & class index
└── results/               # All saved visual outputs
```

### What Are Custom Generators?

A custom generator is a Python function or class that yields batches of data on demand, offering total control over how data is loaded and processed.

You control:

- Disk reading logic (DICOM, TIFF, PNG, custom data)
- Preprocessing sequence
- Augmentation methods
- Batch composition
- Memory management
- Integration with external libraries (OpenCV, PIL, C++)

🔍 When To Use

Use custom generators when:
- Your dataset is non-standard or domain-specific
- You need custom preprocessing / augmentations
- You’re reproducing research pipelines
- You require multiple data sources
- You need total transparency


Avoid them if:

- Standard tools ```(ImageDataGenerator, tf.data)``` suffice
- Simplicity and maintainability are priorities
- You need maximum throughput (prefetching pipelines are faster)

> 💬 “The power of custom generators is flexibility — the cost is maintenance and performance if not optimized carefully.”

## 🧠 Modern Design Note

Although this project conceptually uses a “custom generator,”
it’s implemented with TensorFlow’s tf.data API, providing:

- Parallel I/O and caching
- GPU pipelining
- Native augmentation
- Auto-sharding and batching
- High-speed scalability

⚡ This means you get the control of custom generators with the speed of ```tf.data```.


### 🧰 Commands

🔹 Train the model with full visual pipeline
```python main.py --image_dir /path/to/dataset --epochs 10 --batch_size 32 --augment 1```

🔹 Evaluate the saved model
```python evaluate.py --image_dir /path/to/dataset --rows 2 --cols 5```

🔹 Generate a prediction gallery
```python predict.py --image_dir /path/to/dataset --rows 2 --cols 5```


#### 🧾 Visualization Outputs

Each stage is interactive — close a plot window to see the next.
All results are automatically saved in the ```results/``` folder.

📊 1. Dataset Distribution Graphs

🖼 2. Class-wise Image Grid

🔄 3. Augmentation Preview

📈 4. Training & Validation Curves

🔢 5. Confusion Matrix (Blush Theme)

📜 6. Classification Report

🩺 7. ROC–AUC Curves

🎯 8. Prediction Gallery

Correct = Green ✅ | Wrong = Red ❌ | Confidence shown as %


#### 🧩 Extendability

- You can easily extend this pipeline to:
- Add Grad-CAM / interpretability maps
- Plug into Nextflow or MLflow for experiment tracking
- Replace MobileNetV2 with EfficientNet, ViT, ResNet, etc.
- Export to ONNX / TFLite
- Integrate with Streamlit or Gradio for deployment