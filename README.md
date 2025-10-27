# 🧠 DeepVision-Pipelines

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red.svg)]()

---

### 🔍 Description
> Explore diverse image dataset workflows — from generators to `tf.data` pipelines — designed for deep learning and computer vision projects.

**DeepVision-Pipelines** is a curated collection of image preprocessing and dataset preparation workflows.  
It demonstrates multiple ways to load, preprocess, augment, and batch image data efficiently using:
- **TensorFlow/Keras**
- **PyTorch DataLoader**
- **NumPy/npz based datasets**
- **Custom generators for image pipelines**

The goal is to **help researchers and engineers understand data handling techniques** that directly impact model training performance and accuracy.

---

## 🧩 Key Features
- Multiple dataset loading strategies (`ImageDataGenerator`, `tf.data`, `DataLoader`)
- On-the-fly data augmentation and preprocessing
- Modular and easy-to-extend pipeline design
- Examples compatible with CNN, ViT, and hybrid models
- Well-documented notebooks and visual flow diagrams

---

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
