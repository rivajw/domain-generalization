"""Reconstructed MixStyle + Multi (multi-scale cosine feature consistency).

Source: section IV-H and implementation details of the user's BraTS2023 report.
The original Multi training code was not supplied. This module is a new,
explicitly documented implementation, NOT verified source-code reproduction.

Clean/MixStyle paired features, after MixStyle at L2 and after subsequent L3,
L4 and global average pooling, are compared by per-image cosine distance.
Each of the four terms is weighted equally by w (SUM, not mean).
"""
from __future__ import annotations

from contextlib import contextmanager
import torch
import torch.nn.functional as F
from tqdm import tqdm
from . import runtime
from .consistency import set_all_mixstyle

STAGE_NAMES = ("layer2_postmix", "layer3", "layer4", "pooled")

@contextmanager
def capture_multiscale_features(model):
    """Collect stage outputs via forward hooks without modifying ResNet or state_dict."""
    if model.ms2 is None:
        raise ValueError("MixStyle + Multi requires MixStyle after layer2 (model.ms2).")
    locations = {
        "layer2_postmix": model.ms2,
        "layer3": model.backbone.layer3,
        "layer4": model.backbone.layer4,
        "pooled": model.backbone.avgpool,
    }
    captured = {}
    handles = []
    def make_hook(name):
        def hook(_module, _inp, output):
            captured[name] = output
        return hook
    try:
        for name, module in locations.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()


def multiscale_cosine_loss(mixed, clean):
    """Return sum of four (1 - mean per-image cosine) losses + stage breakdown.

    Spatial feature maps are FLATTENED per example before cosine, not GAP'ed.
    The paper does not specify the reduction method; this is our documented
    reconstruction choice and may differ from the missing original code.
    """
    parts = {}
    for name in STAGE_NAMES:
        if name not in mixed or name not in clean:
            raise KeyError(f"Missing captured stage: {name}")
        student = mixed[name].float().flatten(start_dim=1)
        teacher = clean[name].detach().float().flatten(start_dim=1)
        if student.shape != teacher.shape:
            raise ValueError(f"Shape mismatch at {name}: {student.shape} vs {teacher.shape}")
        parts[name] = (1.0 - F.cosine_similarity(student, teacher, dim=1, eps=1e-8)).mean()
    return sum(parts.values()), parts


def train_epoch_multiscale(model, loader, optimizer, criterion, weight: float = 1.0):
    """One epoch: 2 forward passes, CE + weight * SUM(layer-wise cosine losses).

    The clean teacher pass is no_grad and MixStyle OFF; its features are
    detached. The mixed student pass is differentiable and MixStyle ON.
    Mirrors the training/eval mode and optimizer/scaler handling of the
    supplied dg/training.py's KL consistency routine.
    """
    if weight < 0:
        raise ValueError("multi-scale weight must be nonnegative")
    model.train()
    device, amp = runtime.DEVICE, runtime.USE_AMP
    stats = {"loss": 0., "ce": 0., "multi": 0., "correct": 0, "n": 0, "steps": 0}
    for stage in STAGE_NAMES:
        stats[f"cos_{stage}"] = 0.
    with capture_multiscale_features(model) as features:
        for x, y, _site, _case in tqdm(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                set_all_mixstyle(model, False)
                features.clear()
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                    model(x)
                clean = {name: features[name].detach() for name in STAGE_NAMES}
                features.clear()

                set_all_mixstyle(model, True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    mixed_logits = model(x)
                    ce = criterion(mixed_logits, y)
                    multi, parts = multiscale_cosine_loss(features, clean)
                    loss = ce + weight * multi
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite Multi loss: CE={ce.item()} multi={multi.item()}")
                runtime.scaler.scale(loss).backward()
                runtime.scaler.step(optimizer)
                runtime.scaler.update()
            finally:
                set_all_mixstyle(model, True)
                features.clear()

            stats["loss"] += loss.item()
            stats["ce"] += ce.item()
            stats["multi"] += multi.item()
            for stage, part in parts.items():
                stats[f"cos_{stage}"] += part.item()
            stats["correct"] += (mixed_logits.argmax(dim=1) == y).sum().item()
            stats["n"] += y.numel()
            stats["steps"] += 1
    if not stats["steps"]:
        raise RuntimeError("Empty training loader")
    steps = stats["steps"]
    return {**{k: v / steps for k, v in stats.items() if k not in ("correct", "n", "steps")},
            "train_acc": stats["correct"] / stats["n"]}
