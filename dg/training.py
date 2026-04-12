"""Training and evaluation loops."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from tqdm import tqdm

from . import runtime


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (including CUDA) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y, site, case_id in tqdm(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        runtime.scaler.scale(loss).backward()
        runtime.scaler.step(optimizer)
        runtime.scaler.update()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


@torch.inference_mode()
def eval_epoch(model, loader):
    """Run one eval epoch and return accuracy."""
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.eval()
    correct = 0
    total = 0
    for x, y, site, case_id in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.inference_mode()
def eval_metrics(model, loader):
    """Return a dict with accuracy, balanced accuracy, and tumor P/R/F1."""
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.eval()
    tp = fp = tn = fn = 0
    for x, y, site, case_id in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
        preds = out.argmax(dim=1)
        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        tn += ((preds == 0) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    recall_tumor = tp / (tp + fn) if (tp + fn) else 0.0
    recall_normal = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (recall_tumor + recall_normal)
    precision_tumor = tp / (tp + fp) if (tp + fp) else 0.0
    f1_tumor = (
        (2 * precision_tumor * recall_tumor / (precision_tumor + recall_tumor))
        if (precision_tumor + recall_tumor)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tumor_precision": precision_tumor,
        "tumor_recall": recall_tumor,
        "tumor_f1": f1_tumor,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
