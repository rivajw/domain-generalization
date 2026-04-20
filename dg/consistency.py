"""Consistency analysis and consistency-regularized training for MixStyle.

Phase 1 — Analysis: measure how much MixStyle perturbs predictions on a
trained model by running dual forward passes (MixStyle ON vs OFF).

Phase 2 — Training: add a KL-divergence consistency loss that penalizes
the model when its predictions change under style perturbation.

Phase 3 — Multi-scale: instead of (or in addition to) output-level KL,
compute a feature-space distance (cosine or MSE) between the clean and
mixed representations at several depths. Motivation: if MixStyle is
inserted at layer2, the style perturbation propagates through layer3,
layer4, and the pre-FC features — enforcing consistency at each stage
tests whether intermediate-level consistency matters more than the
logit-level signal that output-only methods (SHADE/CCFP) rely on.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from . import runtime
from .models import MixStyle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_all_mixstyle(model, active: bool) -> None:
    """Toggle every MixStyle module inside *model* without touching train/eval mode."""
    for m in model.modules():
        if isinstance(m, MixStyle):
            m.set_activation_status(active)


# ---------------------------------------------------------------------------
# Phase 1: Consistency measurement on a trained model
# ---------------------------------------------------------------------------
def measure_consistency(model, loader, n_repeats: int = 5):
    """Run dual forward passes (MixStyle OFF vs ON) and collect per-sample stats.

    Because MixStyle is stochastic, the "mixed" pass is repeated *n_repeats*
    times and results are averaged to get a stable estimate.

    Parameters
    ----------
    model : ResNet18MixStyle
        Must have MixStyle modules (i.e. trained with ``use_mixstyle=True``).
    loader : DataLoader
    n_repeats : int
        How many mixed forward passes per batch (averaged for stability).

    Returns
    -------
    list[dict] — one entry per sample with keys:
        clean_pred, clean_conf, mixed_pred_majority, flip, mean_kl,
        mean_conf_delta, label, site
    """
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.train()  # MixStyle only fires in train mode

    records: list[dict] = []

    with torch.no_grad():
        for x, y, sites, case_ids in loader:
            x = x.to(device, non_blocking=True)
            B = x.size(0)

            # --- clean pass (MixStyle OFF) ---
            set_all_mixstyle(model, False)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits_clean = model(x)
            prob_clean = F.softmax(logits_clean, dim=1)
            pred_clean = logits_clean.argmax(dim=1)
            conf_clean = prob_clean.max(dim=1).values

            # --- mixed passes (MixStyle ON, repeated) ---
            set_all_mixstyle(model, True)
            kl_accum = torch.zeros(B, device=device)
            conf_mixed_accum = torch.zeros(B, device=device)
            pred_counts = torch.zeros(B, 2, device=device)  # [B, num_classes]

            for _ in range(n_repeats):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits_mixed = model(x)
                prob_mixed = F.softmax(logits_mixed, dim=1)

                # KL(mixed || clean)  — per-sample
                kl = F.kl_div(
                    prob_mixed.log(), prob_clean, reduction="none"
                ).sum(dim=1)
                kl_accum += kl
                conf_mixed_accum += prob_mixed.max(dim=1).values
                preds_mixed = logits_mixed.argmax(dim=1)
                for c in range(2):
                    pred_counts[:, c] += (preds_mixed == c).float()

            mean_kl = (kl_accum / n_repeats).cpu()
            mean_conf_mixed = (conf_mixed_accum / n_repeats).cpu()
            majority_pred = pred_counts.argmax(dim=1).cpu()
            pred_clean_cpu = pred_clean.cpu()
            conf_clean_cpu = conf_clean.cpu()

            for i in range(B):
                site_val = sites[i] if isinstance(sites[i], int) else int(sites[i])
                records.append({
                    "clean_pred": int(pred_clean_cpu[i]),
                    "clean_conf": float(conf_clean_cpu[i]),
                    "mixed_pred_majority": int(majority_pred[i]),
                    "flip": int(pred_clean_cpu[i] != majority_pred[i]),
                    "mean_kl": float(mean_kl[i]),
                    "mean_conf_delta": float(mean_conf_mixed[i] - conf_clean_cpu[i]),
                    "label": int(y[i]),
                    "site": site_val,
                })

    # restore model to eval + MixStyle active (default state)
    set_all_mixstyle(model, True)
    model.eval()
    return records


def consistency_report(records: list[dict]) -> dict:
    """Print and return a summary of consistency measurement results."""
    n = len(records)
    if n == 0:
        print("No records to summarise.")
        return {}

    flips = sum(r["flip"] for r in records)
    mean_kl = sum(r["mean_kl"] for r in records) / n
    mean_conf_delta = sum(r["mean_conf_delta"] for r in records) / n

    print(f"=== Consistency Report ({n} samples) ===")
    print(f"  Overall flip rate:     {flips}/{n} ({100*flips/n:.2f}%)")
    print(f"  Mean KL(mixed||clean): {mean_kl:.6f}")
    print(f"  Mean confidence delta: {mean_conf_delta:+.4f}")

    # Per-class
    by_class: dict[int, list] = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    print("\n  Per-class breakdown:")
    for cls in sorted(by_class):
        recs = by_class[cls]
        cls_flips = sum(r["flip"] for r in recs)
        cls_kl = sum(r["mean_kl"] for r in recs) / len(recs)
        label = "normal" if cls == 0 else "tumor"
        print(f"    {label} (n={len(recs)}): flip_rate={100*cls_flips/len(recs):.2f}%, mean_kl={cls_kl:.6f}")

    # Per-site
    by_site: dict[int, list] = defaultdict(list)
    for r in records:
        by_site[r["site"]].append(r)

    print("\n  Per-site breakdown:")
    for site in sorted(by_site):
        recs = by_site[site]
        site_flips = sum(r["flip"] for r in recs)
        site_kl = sum(r["mean_kl"] for r in recs) / len(recs)
        print(f"    site {site:02d} (n={len(recs)}): flip_rate={100*site_flips/len(recs):.2f}%, mean_kl={site_kl:.6f}")

    return {
        "n": n,
        "flip_rate": flips / n,
        "mean_kl": mean_kl,
        "mean_conf_delta": mean_conf_delta,
    }


# ---------------------------------------------------------------------------
# Phase 3: Multi-scale feature-space consistency
# ---------------------------------------------------------------------------
_VALID_MS_KEYS = ("layer1", "layer2", "layer3", "layer4", "feats")


def _gap(x: torch.Tensor) -> torch.Tensor:
    """Spatially pool a feature map to (B, C). 2D tensors pass through."""
    if x.ndim == 4:
        return x.mean(dim=(2, 3))
    if x.ndim == 2:
        return x
    return x.flatten(1)


def multiscale_feature_consistency(
    inter_mixed: dict,
    inter_clean: dict,
    layer_weights: dict,
    distance: str = "cosine",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted sum of feature-space distances between clean and mixed branches.

    Clean features are assumed to be already detached (stop-gradient teacher).
    Spatial maps are reduced via GAP so every layer contributes a (B, C) tensor.

    Parameters
    ----------
    inter_mixed, inter_clean
        Dicts keyed by "layer1".."layer4" and "feats" (from forward(..., return_multiscale=True)).
    layer_weights
        Dict mapping stage name → non-negative weight. Stages with weight 0 or
        missing are skipped.
    distance
        "cosine" → 1 - cosine_sim (per-sample, averaged). Scale-invariant, only
        direction is matched — a good default when we still want the mixed
        branch to shift magnitude.
        "mse" → mean squared error per channel. Penalises scale drift too.
    """
    if distance not in ("cosine", "mse"):
        raise ValueError(f"unknown distance: {distance!r}")

    total = inter_mixed[next(iter(layer_weights))].new_zeros(())
    any_active = False
    for key, w in layer_weights.items():
        if w <= 0:
            continue
        if key not in _VALID_MS_KEYS:
            raise ValueError(f"unknown stage {key!r}; expected one of {_VALID_MS_KEYS}")
        fm = _gap(inter_mixed[key])
        fc = _gap(inter_clean[key]).detach()
        if distance == "cosine":
            fm_n = F.normalize(fm, dim=1, eps=eps)
            fc_n = F.normalize(fc, dim=1, eps=eps)
            d = 1.0 - (fm_n * fc_n).sum(dim=1)
        else:  # mse
            d = (fm - fc).pow(2).mean(dim=1)
        total = total + float(w) * d.mean()
        any_active = True

    if not any_active:
        return inter_mixed["feats"].new_zeros(())
    return total
