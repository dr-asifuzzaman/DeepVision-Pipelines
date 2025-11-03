# data_loader.py
import json
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import StratifiedShuffleSplit

from config import IMG_SIZE, NUM_WORKERS, VAL_SIZE, TEST_SIZE, SEED, SPLIT_JSON, CLASS_MAP_JSON


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def train_transform(augment: bool, img_size=IMG_SIZE):
    if not augment:
        return base_transform(img_size)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])


class TransformingSubset(Subset):
    """Subset that applies its own transform by re-loading from the file path."""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, target = self.dataset.samples[real_idx]
        img = self.dataset.loader(path)  # PIL image
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)


def stratified_split(y: np.ndarray, val_size: float, test_size: float, seed: int = SEED):
    indices = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, y))

    temp_y = y[temp_idx]
    val_fraction_of_temp = val_size / (val_size + test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_fraction_of_temp), random_state=seed)
    val_rel, test_rel = next(sss2.split(np.arange(len(temp_idx)), temp_y))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


def make_dataloaders(
    image_dir: str,
    batch_size: int,
    augment: bool,
    img_size: int = IMG_SIZE,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns train/val/test dataloaders + info dict with class_names and split indices.
    """
    set_seed(seed)

    image_dir = str(image_dir)
    dataset = datasets.ImageFolder(root=image_dir, transform=base_transform(img_size))
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx

    # Save mapping for later use
    CLASS_MAP_JSON.write_text(json.dumps({"class_to_idx": class_to_idx, "classes": class_names}, indent=2))

    targets = np.array([t for _, t in dataset.samples])

    train_idx, val_idx, test_idx = stratified_split(
        y=targets, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=seed
    )

    # Save splits for reproducibility
    SPLIT_JSON.write_text(json.dumps({
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist()
    }, indent=2))

    tr_ds = TransformingSubset(dataset, train_idx, transform=train_transform(bool(augment), img_size))
    va_ds = TransformingSubset(dataset, val_idx,   transform=base_transform(img_size))
    te_ds = TransformingSubset(dataset, test_idx,  transform=base_transform(img_size))

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    info = dict(
        class_names=class_names,
        class_to_idx=class_to_idx,
        indices=dict(train=train_idx, val=val_idx, test=test_idx),
        dataset=dataset  # for access to file paths if needed
    )
    return train_loader, val_loader, test_loader, info
