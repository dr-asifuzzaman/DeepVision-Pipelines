# DeepVision-Pipelines

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red.svg)]()

---

### Description
> Explore diverse image dataset workflows — from generators to `tf.data` pipelines — designed for deep learning and computer vision projects.

**DeepVision-Pipelines** is a curated collection of image preprocessing and dataset preparation workflows.  
It demonstrates multiple ways to load, preprocess, augment, and batch image data efficiently using:
- **TensorFlow/Keras**
- **PyTorch DataLoader**
- **NumPy/npz based datasets**
- **Custom generators for image pipelines**

The goal is to **help researchers and engineers understand data handling techniques** that directly impact model training performance and accuracy.

##### **Keywords:** 
`deep-learning`, `computer-vision`, `image-processing`, `tensorflow`, `pytorch`, `tfdata`, `tfrrecord`, `dataloader`, keras, `dataset-pipeline`
---

## Key Features
- Multiple dataset loading strategies (`ImageDataGenerator`, `tf.data`, `DataLoader`)
- On-the-fly data augmentation and preprocessing
- Modular and easy-to-extend pipeline design
- Examples compatible with CNN, ViT, and hybrid models
- Well-documented notebooks and visual flow diagrams

---
<!--- 
## 🧱 Folder Structure

DeepVision-Pipelines/
│
├── 01_flow_from_directory/
│   ├── dataset/ (sample folders)
│   ├── flow_from_directory_demo.ipynb
│   ├── README.md
│
├── 02_flow_from_dataframe/
│   ├── dataset/ (images + CSV)
│   ├── flow_from_dataframe_demo.ipynb
│   ├── README.md
│
├── 03_npz_numpy_pipeline/
│   ├── save_npz_dataset.ipynb
│   ├── load_npz_pipeline.ipynb
│
├── 04_tf_data_pipeline/
│   ├── tf_data_demo.ipynb
│   ├── README.md
│
├── 05_tfrecord_pipeline/
│   ├── create_tfrecord.ipynb
│   ├── read_tfrecord.ipynb
│
├── 06_custom_generator/
│   ├── custom_generator_demo.ipynb
│   ├── README.md
│
├── 07_pytorch_dataloader/
│   ├── pytorch_loader_demo.ipynb
│   ├── README.md
│
├── 08_augmentation_pipeline/
│   ├── augmentation_demo.ipynb
│   ├── README.md
│
├── assets/               # images, logos, visual examples
├── utils/                # common functions, configs
├── README.md             # main showcase page
├── requirements.txt
└── LICENSE


---
--->

## Example Pipelines Overview

| Pipeline Type | Framework | Description | Notebook |
|----------------|------------|--------------|-----------|
| Image Generator | TensorFlow/Keras | Classical image loading with augmentation | `tensorflow_pipelines/image_generator.ipynb` |
| `tf.data` Pipeline | TensorFlow | High-performance pipeline using `tf.data.Dataset` | `tensorflow_pipelines/tfdata_pipeline.ipynb` |
| Custom DataLoader | PyTorch | Data loading and transforms using PyTorch | `pytorch_pipelines/dataloader_pipeline.ipynb` |
| NumPy-based | NumPy | Creating and reading `.npz` dataset files | `numpy_pipelines/npz_loader.ipynb` |
| Comparison | Mixed | Compare TensorFlow vs PyTorch pipelines | `examples/compare_pipelines.ipynb` |

---

## Getting Started

### Clone this repository
git clone https://github.com/<your-username>/DeepVision-Pipelines.git
cd DeepVision-Pipelines

###  Install dependencies
pip install -r requirements.txt


---


**Use when:** You want to expand your dataset artificially

---

## **Quick Decision Guide**

### Choose your pipeline based on:

| Your Situation | Best Pipeline |
|----------------|---------------|
| Just starting out | `01_flow_from_directory` |
| Have organized spreadsheet | `02_flow_from_dataframe` |
| Small dataset, need speed | `03_npz_numpy` |
| Using TensorFlow professionally | `04_tf_data` |
| MASSIVE dataset (ImageNet-scale) | `05_tfrecord` |
| Special requirements | `06_custom_generator` |
| Using PyTorch | `07_pytorch_dataloader` |
| Need data variety | `08_augmentation` |

---

###  **How They All Work Together**
1. Pick a pipeline method (folders 01-07)
2. Add augmentation if needed (folder 08)
3. Feed to your AI model
4. Train and get results!
