"""Training and evaluation loops."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import runtime
from .mmd import compute_batch_mmd_loss, forward_with_features
from .models import configure_all_mixstyle, set_all_mixstyle


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


def _safe_epoch_stats(total_loss: float, correct: int, total: int, num_valid_batches: int):
    if num_valid_batches == 0:
        return float("nan"), 0.0 if total == 0 else correct / total
    return total_loss / num_valid_batches, 0.0 if total == 0 else correct / total


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


def train_epoch_consistency(
    model,
    loader,
    optimizer,
    criterion,
    lambda_c: float = 0.1,
    mixstyle_enabled: bool = True,
    mixstyle_p: Optional[float] = None,
    mixstyle_a: Optional[float] = None,
):
    """Training epoch with consistency regularization.

    For each batch:
    1. Clean forward pass (MixStyle OFF) → logits_clean (detached target)
    2. Mixed forward pass (MixStyle schedule applied) → logits_mixed
    3. loss = CE(logits_mixed, y) + lambda_c * KL(mixed_softmax || clean_softmax.detach())

    Returns (avg_loss, avg_ce_loss, avg_kl_loss, accuracy).
    """
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct = 0
    total = 0
    num_valid_batches = 0

    for x, y, site, case_id in tqdm(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        configure_all_mixstyle(model, enabled=False)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits_clean = model(x)
        prob_clean = F.softmax(logits_clean, dim=1).detach()

        configure_all_mixstyle(
            model,
            enabled=mixstyle_enabled,
            p=mixstyle_p,
            a=mixstyle_a,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits_mixed = model(x)
            ce_loss = criterion(logits_mixed, y)

            log_prob_mixed = F.log_softmax(logits_mixed, dim=1)
            kl_loss = F.kl_div(log_prob_mixed, prob_clean, reduction="batchmean")

            loss = ce_loss + lambda_c * kl_loss

        runtime.scaler.scale(loss).backward()
        runtime.scaler.step(optimizer)
        runtime.scaler.update()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        preds = logits_mixed.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        num_valid_batches += 1

    if num_valid_batches == 0:
        return float("nan"), float("nan"), float("nan"), 0.0

    return (
        total_loss / num_valid_batches,
        total_ce / num_valid_batches,
        total_kl / num_valid_batches,
        correct / total if total > 0 else 0.0,
    )


def train_epoch_mmd(
    model,
    loader,
    optimizer,
    criterion,
    lambda_mmd: float = 0.1,
    class_conditional: bool = True,
    sigma_list: tuple[float, ...] = (1.0, 5.0, 10.0),
    max_samples: int | None = 256,
    mixstyle_enabled: bool = True,
    mixstyle_p: Optional[float] = None,
    mixstyle_a: Optional[float] = None,
):
    """Training epoch with MMD regularization.

    loss = CE(logits, y) + lambda_mmd * MMD(features across sites)

    Returns
    -------
    (avg_loss, avg_ce_loss, avg_mmd_loss, accuracy)
    """
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_mmd = 0.0
    correct = 0
    total = 0
    num_valid_batches = 0

    for x, y, site, case_id in tqdm(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        site = site.to(device, non_blocking=True)

        configure_all_mixstyle(
            model,
            enabled=mixstyle_enabled,
            p=mixstyle_p,
            a=mixstyle_a,
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, feats = forward_with_features(model, x)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("LOGITS NAN/INF")
                continue

            if torch.isnan(feats).any() or torch.isinf(feats).any():
                print("FEATURES NAN/INF")
                continue

            ce_loss = criterion(logits, y)
            if lambda_mmd > 0:
                mmd_loss = compute_batch_mmd_loss(
                    feats,
                    site,
                    labels=y,
                    class_conditional=class_conditional,
                    sigma_list=sigma_list,
                    max_samples=max_samples,
                )
            else:
                mmd_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
            loss = ce_loss + lambda_mmd * mmd_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("LOSS IS NAN/INF")
            print(f"ce_loss={ce_loss.item()}, mmd_loss={mmd_loss.item()}")
            continue

        runtime.scaler.scale(loss).backward()
        runtime.scaler.step(optimizer)
        runtime.scaler.update()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_mmd += mmd_loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        num_valid_batches += 1

    if num_valid_batches == 0:
        return float("nan"), float("nan"), float("nan"), 0.0

    return (
        total_loss / num_valid_batches,
        total_ce / num_valid_batches,
        total_mmd / num_valid_batches,
        correct / total if total > 0 else 0.0,
    )


def train_epoch_consistency_mmd(
    model,
    loader,
    optimizer,
    criterion,
    lambda_c: float = 0.1,
    lambda_mmd: float = 0.1,
    class_conditional: bool = True,
    sigma_list: tuple[float, ...] = (1.0, 5.0, 10.0),
    max_samples: int | None = 256,
    mixstyle_enabled: bool = True,
    mixstyle_p: Optional[float] = None,
    mixstyle_a: Optional[float] = None,
    mmd_on_clean_features: bool = False,
):
    """Training epoch with both consistency regularization and MMD regularization.

    loss = CE(logits_mixed, y) + lambda_c * KL(mixed_softmax || clean_softmax.detach())
            + lambda_mmd * MMD(features across sites)

    Returns
    -------
    (avg_loss, avg_ce_loss, avg_kl_loss, avg_mmd_loss, accuracy)
    """
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_mmd = 0.0
    correct = 0
    total = 0
    num_valid_batches = 0

    for x, y, site, case_id in tqdm(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        site = site.to(device, non_blocking=True)

        configure_all_mixstyle(model, enabled=False)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits_clean, feats_clean = forward_with_features(model, x)
        prob_clean = F.softmax(logits_clean, dim=1).detach()

        configure_all_mixstyle(
            model,
            enabled=mixstyle_enabled,
            p=mixstyle_p,
            a=mixstyle_a,
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits_mixed, feats_mixed = forward_with_features(model, x)

            if torch.isnan(logits_mixed).any() or torch.isinf(logits_mixed).any():
                print("LOGITS NAN/INF")
                continue

            if torch.isnan(feats_mixed).any() or torch.isinf(feats_mixed).any():
                print("FEATURES NAN/INF")
                continue

            ce_loss = criterion(logits_mixed, y)

            log_prob_mixed = F.log_softmax(logits_mixed, dim=1)
            kl_loss = F.kl_div(log_prob_mixed, prob_clean, reduction="batchmean")

            if lambda_mmd > 0:
                feats_for_mmd = feats_clean if mmd_on_clean_features else feats_mixed
                mmd_loss = compute_batch_mmd_loss(
                    feats_for_mmd,
                    site,
                    labels=y,
                    class_conditional=class_conditional,
                    sigma_list=sigma_list,
                    max_samples=max_samples,
                )
            else:
                mmd_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)

            loss = ce_loss + lambda_c * kl_loss + lambda_mmd * mmd_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print("LOSS IS NAN/INF")
            print(
                f"ce_loss={ce_loss.item()}, "
                f"kl_loss={kl_loss.item()}, "
                f"mmd_loss={mmd_loss.item()}"
            )
            continue

        runtime.scaler.scale(loss).backward()
        runtime.scaler.step(optimizer)
        runtime.scaler.update()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        total_mmd += mmd_loss.item()

        preds = logits_mixed.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        num_valid_batches += 1

    if num_valid_batches == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0

    return (
        total_loss / num_valid_batches,
        total_ce / num_valid_batches,
        total_kl / num_valid_batches,
        total_mmd / num_valid_batches,
        correct / total if total > 0 else 0.0,
    )


@torch.inference_mode()
def eval_epoch(model, loader):
    """Run one eval epoch and return accuracy."""
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    configure_all_mixstyle(model, enabled=False)
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

    configure_all_mixstyle(model, enabled=False)
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
