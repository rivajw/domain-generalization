"""MMD utilities for MixStyle-based domain generalization."""

from __future__ import annotations

from collections import defaultdict

import torch

from . import runtime
from .consistency import set_all_mixstyle


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def forward_with_features(model, x: torch.Tensor):
    """Return (logits, features). Requires model(x, return_features=True)."""
    return model(x, return_features=True)


def _flatten_features(z: torch.Tensor) -> torch.Tensor:
    if z.ndim > 2:
        z = z.flatten(1)
    return z


def _subsample_rows(x: torch.Tensor, max_rows: int | None) -> torch.Tensor:
    if max_rows is None or x.size(0) <= max_rows:
        return x
    idx = torch.randperm(x.size(0), device=x.device)[:max_rows]
    return x[idx]


# ---------------------------------------------------------------------------
# MMD core
# ---------------------------------------------------------------------------
def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return d2.clamp_min(0.0)


def _rbf_kernel_sum(d2: torch.Tensor, sigma_list: tuple[float, ...]) -> torch.Tensor:
    out = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * sigma * sigma)
        out = out + torch.exp(-gamma * d2)
    return out


def gaussian_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_list: tuple[float, ...] = (1.0, 5.0, 10.0),
    max_samples: int | None = 256,
) -> torch.Tensor:
    """Biased multi-kernel MMD^2 estimate."""
    x = _flatten_features(x)
    y = _flatten_features(y)

    x = _subsample_rows(x, max_samples)
    y = _subsample_rows(y, max_samples)

    if x.size(0) < 2 or y.size(0) < 2:
        return torch.tensor(0.0, device=x.device)

    dxx = _pairwise_sq_dists(x, x)
    dyy = _pairwise_sq_dists(y, y)
    dxy = _pairwise_sq_dists(x, y)

    kxx = _rbf_kernel_sum(dxx, sigma_list).mean()
    kyy = _rbf_kernel_sum(dyy, sigma_list).mean()
    kxy = _rbf_kernel_sum(dxy, sigma_list).mean()

    return kxx + kyy - 2.0 * kxy


def compute_batch_mmd_loss(
    features: torch.Tensor,
    sites: torch.Tensor,
    labels: torch.Tensor | None = None,
    class_conditional: bool = True,
    sigma_list: tuple[float, ...] = (1.0, 5.0, 10.0),
    max_samples: int | None = 256,
) -> torch.Tensor:
    """Compute site-alignment MMD within one batch."""
    features = _flatten_features(features)
    unique_sites = torch.unique(sites)

    if unique_sites.numel() < 2:
        return torch.tensor(0.0, device=features.device)

    losses = []

    if class_conditional:
        if labels is None:
            raise ValueError("labels must be provided when class_conditional=True")

        unique_labels = torch.unique(labels)

        for i in range(unique_sites.numel()):
            for j in range(i + 1, unique_sites.numel()):
                si = unique_sites[i]
                sj = unique_sites[j]
                for c in unique_labels:
                    fi = features[(sites == si) & (labels == c)]
                    fj = features[(sites == sj) & (labels == c)]
                    if fi.size(0) >= 2 and fj.size(0) >= 2:
                        losses.append(
                            gaussian_mmd(
                                fi,
                                fj,
                                sigma_list=sigma_list,
                                max_samples=max_samples,
                            )
                        )
    else:
        for i in range(unique_sites.numel()):
            for j in range(i + 1, unique_sites.numel()):
                si = unique_sites[i]
                sj = unique_sites[j]
                fi = features[sites == si]
                fj = features[sites == sj]
                if fi.size(0) >= 2 and fj.size(0) >= 2:
                    losses.append(
                        gaussian_mmd(
                            fi,
                            fj,
                            sigma_list=sigma_list,
                            max_samples=max_samples,
                        )
                    )

    if not losses:
        return torch.tensor(0.0, device=features.device)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def measure_mmd_alignment(
    model,
    loader,
    n_repeats: int = 5,
    class_conditional: bool = True,
    sigma_list: tuple[float, ...] = (1.0, 5.0, 10.0),
    max_samples: int | None = 256,
):
    """Compare clean-vs-mixed cross-site MMD on a loader.

    Best used on the TRAIN loader, since it contains multiple sites.
    """
    device = runtime.DEVICE
    use_amp = runtime.USE_AMP

    model.train()
    records: list[dict] = []

    with torch.no_grad():
        for x, y, sites, case_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sites = sites.to(device, non_blocking=True)

            unique_sites = torch.unique(sites)
            if unique_sites.numel() < 2:
                continue

            # Clean features
            set_all_mixstyle(model, False)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _, feats_clean = forward_with_features(model, x)

            # Mixed features averaged across repeats
            set_all_mixstyle(model, True)
            mixed_list = []
            for _ in range(n_repeats):
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    _, feats_mixed = forward_with_features(model, x)
                mixed_list.append(feats_mixed)
            feats_mixed = torch.stack(mixed_list, dim=0).mean(dim=0)

            if class_conditional:
                unique_labels = torch.unique(y)
                for i in range(unique_sites.numel()):
                    for j in range(i + 1, unique_sites.numel()):
                        si = unique_sites[i]
                        sj = unique_sites[j]
                        for c in unique_labels:
                            fi_clean = feats_clean[(sites == si) & (y == c)]
                            fj_clean = feats_clean[(sites == sj) & (y == c)]
                            fi_mixed = feats_mixed[(sites == si) & (y == c)]
                            fj_mixed = feats_mixed[(sites == sj) & (y == c)]

                            if fi_clean.size(0) >= 2 and fj_clean.size(0) >= 2:
                                clean_mmd = gaussian_mmd(
                                    fi_clean,
                                    fj_clean,
                                    sigma_list=sigma_list,
                                    max_samples=max_samples,
                                )
                                mixed_mmd = gaussian_mmd(
                                    fi_mixed,
                                    fj_mixed,
                                    sigma_list=sigma_list,
                                    max_samples=max_samples,
                                )
                                records.append({
                                    "site_i": int(si.item()),
                                    "site_j": int(sj.item()),
                                    "label": int(c.item()),
                                    "clean_mmd": float(clean_mmd.item()),
                                    "mixed_mmd": float(mixed_mmd.item()),
                                    "delta_mmd": float((mixed_mmd - clean_mmd).item()),
                                })
            else:
                for i in range(unique_sites.numel()):
                    for j in range(i + 1, unique_sites.numel()):
                        si = unique_sites[i]
                        sj = unique_sites[j]
                        fi_clean = feats_clean[sites == si]
                        fj_clean = feats_clean[sites == sj]
                        fi_mixed = feats_mixed[sites == si]
                        fj_mixed = feats_mixed[sites == sj]

                        if fi_clean.size(0) >= 2 and fj_clean.size(0) >= 2:
                            clean_mmd = gaussian_mmd(
                                fi_clean,
                                fj_clean,
                                sigma_list=sigma_list,
                                max_samples=max_samples,
                            )
                            mixed_mmd = gaussian_mmd(
                                fi_mixed,
                                fj_mixed,
                                sigma_list=sigma_list,
                                max_samples=max_samples,
                            )
                            records.append({
                                "site_i": int(si.item()),
                                "site_j": int(sj.item()),
                                "label": None,
                                "clean_mmd": float(clean_mmd.item()),
                                "mixed_mmd": float(mixed_mmd.item()),
                                "delta_mmd": float((mixed_mmd - clean_mmd).item()),
                            })

    set_all_mixstyle(model, True)
    model.eval()
    return records


def mmd_alignment_report(records: list[dict], title: str = "MMD Alignment") -> dict:
    if len(records) == 0:
        print(f"=== {title} ===")
        print("No records to summarise.")
        return {}

    n = len(records)
    mean_clean = sum(r["clean_mmd"] for r in records) / n
    mean_mixed = sum(r["mixed_mmd"] for r in records) / n
    mean_delta = sum(r["delta_mmd"] for r in records) / n

    print(f"=== {title} ({n} records) ===")
    print(f"  Mean clean MMD: {mean_clean:.6f}")
    print(f"  Mean mixed MMD: {mean_mixed:.6f}")
    print(f"  Mean delta MMD (mixed-clean): {mean_delta:+.6f}")

    by_class: dict[int, list] = defaultdict(list)
    for r in records:
        if r["label"] is not None:
            by_class[r["label"]].append(r)

    if by_class:
        print("\n  Per-class breakdown:")
        for cls in sorted(by_class):
            recs = by_class[cls]
            c_clean = sum(r["clean_mmd"] for r in recs) / len(recs)
            c_mixed = sum(r["mixed_mmd"] for r in recs) / len(recs)
            c_delta = sum(r["delta_mmd"] for r in recs) / len(recs)
            label_name = "normal" if cls == 0 else "tumor"
            print(
                f"    {label_name} (n={len(recs)}): "
                f"clean={c_clean:.6f}, mixed={c_mixed:.6f}, delta={c_delta:+.6f}"
            )

    return {
        "n": n,
        "mean_clean_mmd": mean_clean,
        "mean_mixed_mmd": mean_mixed,
        "mean_delta_mmd": mean_delta,
    }