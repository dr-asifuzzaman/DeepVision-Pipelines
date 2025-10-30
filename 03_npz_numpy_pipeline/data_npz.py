
import os
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import psutil

from config import IMAGE_DIR, NPZ_PATH, STRUCTURED_NPZ_PATH, TARGET_SIZE, TEST_SIZE, VAL_SIZE_WITHIN_TEST, RANDOM_STATE

def load_and_preprocess_images(image_dir, target_size=(224, 224)):
    images, labels = [], []
    for class_name in sorted(os.listdir(image_dir)):
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
    return np.array(images), np.array(labels)

def save_npz(X, y, path):
    np.savez_compressed(path, images=X, labels=y)
    print(f"Saved NPZ to {path} with {len(X)} samples.")

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data['images'], data['labels']

def split_and_encode(X, y, test_size=0.3, val_size_within=0.5, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size_within, stratify=y_temp, random_state=seed
    )
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    class_names = le.classes_
    return (X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, class_names)

def memory_report(npz_path):
    file_size_gb = os.path.getsize(npz_path) / (1024**3)
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB | Current RAM usage: {memory_gb:.2f} GB")

def save_structured_npz(X_train, X_val, X_test, y_train, y_val, y_test, class_names, path=STRUCTURED_NPZ_PATH):
    info = {
        'version': '1.0',
        'created': 'auto',
        'num_samples': int(len(X_train) + len(X_val) + len(X_test)),
        'preprocessing': 'resize_224_normalize',
        'class_names': list(class_names)
    }
    np.savez_compressed(
        path,
        train_images=X_train,
        train_labels=y_train,
        val_images=X_val,
        val_labels=y_val,
        test_images=X_test,
        test_labels=y_test,
        info=info
    )
    print(f"Saved structured dataset to {path}")

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default=IMAGE_DIR)
    parser.add_argument('--npz_path', default=NPZ_PATH)
    parser.add_argument('--create_npz', type=int, default=1)
    parsed = parser.parse_args(args)

    if parsed.create_npz:
        X, y = load_and_preprocess_images(parsed.image_dir, TARGET_SIZE)
        print(f"Loaded {len(X)} images from {parsed.image_dir}")
        save_npz(X, y, parsed.npz_path)
        memory_report(parsed.npz_path)
    else:
        X, y = load_npz(parsed.npz_path)
        print(f"Loaded {len(X)} images from {parsed.npz_path}")

    # Always show a quick summary
    print("Unique classes:", np.unique(y))

if __name__ == '__main__':
    main()
