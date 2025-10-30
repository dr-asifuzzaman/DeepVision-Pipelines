
# NPZ Image Pipeline (Modular)

A clean, modular pipeline that turns an image folder into a compressed `.npz` dataset, 
trains a CNN, and provides rich visualizations & evaluation:

- Dataset distribution graphs
- Class-wise image grids
- Augmentation previews
- Training & validation curves (side-by-side)
- Confusion matrix (with color map)
- Classification report
- ROC–AUC (all classes in a single plot + micro/macro)
- Prediction gallery with labels & confidence (misclassifications in red)

## Structure

```
npz_pipeline/
├── README.md
├── requirements.txt
├── config.py
├── data_npz.py
├── viz.py
├── train.py
├── evaluate.py
├── predict.py
└── main.py
```

Each module can run **independently** for quick testing (`python module.py`), 
and `main.py` orchestrates the full pipeline.

## Quickstart

1) (Optional) Create a venv and install requirements
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Set your dataset path in `config.py` (folder layout: `root/class/*.jpg|png|jpeg`).

3) End-to-end run (create NPZ → split/encode → train → evaluate → plots):
```bash
python main.py --image_dir /path/to/dataset --npz_path dataset_compressed.npz   --epochs 10 --batch_size 32 --augment 1
```

### Examples

- Create NPZ only:
```bash
python data_npz.py --image_dir ./dataset --npz_path ./dataset_compressed.npz --create_npz 1
```

- Train only (expects NPZ exists):
```bash
python train.py --npz_path ./dataset_compressed.npz --epochs 5 --augment 1
```

- Evaluate + Plots:
```bash
python evaluate.py --npz_path ./dataset_compressed.npz
```

- Prediction gallery:
```bash
python predict.py --npz_path ./dataset_compressed.npz --rows 2 --cols 5
```

> All plots use matplotlib. No seaborn. One figure per chart.
