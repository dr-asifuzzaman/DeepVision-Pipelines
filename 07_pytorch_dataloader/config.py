# config.py
import os
from pathlib import Path

# Defaults (overridden by CLI)
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 10
LR = 1e-3
PATIENCE = 5
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
AUGMENT = 1  # 1=True, 0=False

# Folders
ARTIFACTS_DIR = Path("model_artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

for p in [ARTIFACTS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(p, exist_ok=True)

# Filenames
BEST_CKPT = CHECKPOINTS_DIR / "best_model.pt"
CLASS_MAP_JSON = ARTIFACTS_DIR / "class_mapping.json"
HISTORY_JSON = ARTIFACTS_DIR / "train_history.json"
SPLIT_JSON = ARTIFACTS_DIR / "split_indices.json"
