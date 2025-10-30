
# Central configuration

IMAGE_DIR = "./dataset"              # change to your dataset folder
NPZ_PATH = "./dataset_compressed.npz"
STRUCTURED_NPZ_PATH = "./dataset_full_v1.npz"

TARGET_SIZE = (224, 224)
TEST_SIZE = 0.3
VAL_SIZE_WITHIN_TEST = 0.5           # split of temp into val/test
RANDOM_STATE = 42

# Training
BATCH_SIZE = 32
EPOCHS = 10
AUGMENT = True

# Augmentation params (used if AUGMENT True)
AUG_PARAMS = dict(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
