# config.py
import os
import random
import numpy as np
import tensorflow as tf

# ---------- Defaults (CLI can override) ----------
DATA_ROOT   = "dataset_train_test_val"   # expects train/val/test subdirs
SINGLE_ROOT = None                       # if you only have one root with class folders

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 1337
VAL_SPLIT = 0.2
TEST_FRACTION_OF_VAL = 0.5

EPOCHS = 30
BASE_LR = 1e-3
FINE_TUNE_AT = None            # e.g. 100

CACHE_IN_MEMORY = True
DETERMINISTIC = True

MODEL_DIR = "model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Environment setup ----------
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

AUTOTUNE = tf.data.AUTOTUNE
PREFETCH = AUTOTUNE
