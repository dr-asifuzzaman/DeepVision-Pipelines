# tf_record_pipeline/config.py
from pathlib import Path

# ====== Core training config ======
IMG_SIZE = (224, 224)
CHANNELS = 3
NUM_WORKERS =  tf_AUTOTUNE = None  # will be set in code to tf.data.AUTOTUNE
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
SHUFFLE_BUFFER = 4096
USE_MIXED_PRECISION = False

# ====== TFRecord settings ======
USE_GZIP_TFRECORDS = True
RECORDS_PER_SHARD = 2000

# ====== Model / Artifacts ======
ARTIFACTS_DIR = Path("model_artifacts")
TFRECORD_DIR = ARTIFACTS_DIR / "tfrecords"
BEST_MODEL_PATH = ARTIFACTS_DIR / "model_best.keras"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "model_final.keras"
LABELS_JSON = ARTIFACTS_DIR / "labels.json"
TRAIN_LOG_CSV = ARTIFACTS_DIR / "train_log.csv"

# ====== Visualization defaults ======
GRID_COLS = 5     # class-wise grid columns
AUG_SAMPLES = 6   # augmentation previews per image
