# train_utils.py
import json
import time
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import BEST_CKPT, HISTORY_JSON, PATIENCE


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < (self.best - self.min_delta):
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_fit(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device,
) -> Dict[str, list]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True
    )
    early = EarlyStopping(patience=PATIENCE)
    best_val = np.inf

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss);   history["val_acc"].append(val_acc)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict()}, BEST_CKPT)

        early.step(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | ")

        if early.should_stop:
            print("Early stopping triggered.")
            break

    # persist history
    with open(HISTORY_JSON, "w") as f:
        json.dump(history, f, indent=2)

    # load best weights
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    return history


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels, all_imgs = [], [], [], []
    softmax = nn.Softmax(dim=1)
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        probs = softmax(logits).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_imgs.append(imgs)  # keep CPU tensors for gallery
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        torch.cat(all_imgs, dim=0)
    )
