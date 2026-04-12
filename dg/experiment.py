"""High-level experiment runners: configs, model builder, and LOSO + variant loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import config, runtime
from .dataset import BraTSNPZSliceDataset
from .models import ResNet18MixStyle
from .splits import make_loso_split
from .training import eval_epoch, eval_metrics, set_seed, train_epoch


# ---------------------------------------------------------------------------
# Data / model configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataSplitConfig:
    name: str
    train_cases: FrozenSet[str]
    validation_cases: FrozenSet[str]
    test_cases: FrozenSet[str]
    out_domain_test_cases: FrozenSet[str] = frozenset()

    def prepare_dataloaders(self, index_df: pd.DataFrame, batch_size: int = 128):
        train_df = index_df[index_df["case_id"].isin(self.train_cases)].copy()
        val_df = index_df[index_df["case_id"].isin(self.validation_cases)].copy()
        test_df = index_df[index_df["case_id"].isin(self.test_cases)].copy()
        out_df = index_df[index_df["case_id"].isin(self.out_domain_test_cases)].copy()

        train_ds = BraTSNPZSliceDataset(train_df["npz_path"].tolist())
        val_ds = BraTSNPZSliceDataset(val_df["npz_path"].tolist())
        test_ds = BraTSNPZSliceDataset(test_df["npz_path"].tolist())
        out_ds = BraTSNPZSliceDataset(out_df["npz_path"].tolist())

        dl_kwargs = dict(num_workers=0, pin_memory=True, persistent_workers=False)
        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "out_domain_test_df": out_df,
            "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs),
            "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs),
            "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs),
            "out_domain_test_loader": DataLoader(out_ds, batch_size=batch_size, shuffle=False, **dl_kwargs),
        }


def prepare_dataloaders_from_config(
    data_cfg: DataSplitConfig,
    index_df: pd.DataFrame,
    batch_size: int = 128,
    verbose: bool = True,
):
    loaders = data_cfg.prepare_dataloaders(index_df=index_df, batch_size=batch_size)
    if verbose:
        print(f"\nData Config: {data_cfg.name}")
        for split in ("train", "val", "test", "out_domain_test"):
            df = loaders[f"{split}_df"]
            print(
                f"  {split} cases={df['case_id'].nunique()} "
                f"slices={int(df['num_slices'].sum())} "
                f"sites={sorted(df['site'].unique().tolist())}"
            )
    return loaders


@dataclass(frozen=True)
class ModelConfig:
    name: str
    train_layers: tuple[int, ...] = tuple()
    use_mixstyle: bool = False
    mixstyle_p: float = 0.5
    mixstyle_a: float = 0.1


def build_model(cfg: ModelConfig, pretrained: bool = True, verbose: bool = True):
    insert_after = tuple(f"layer{layer}" for layer in cfg.train_layers) if cfg.use_mixstyle else tuple()
    model = ResNet18MixStyle(
        num_classes=2,
        pretrained=pretrained,
        mixstyle_p=cfg.mixstyle_p,
        mixstyle_alpha=cfg.mixstyle_a,
        insert_after=insert_after,
    ).to(runtime.DEVICE)

    # Freeze all, then unfreeze only the selected layers + fc
    for p in model.parameters():
        p.requires_grad = False
    for layer in cfg.train_layers:
        getattr(model.backbone, f"layer{layer}").requires_grad_(True)
    model.backbone.fc.requires_grad_(True)

    if verbose:
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        layer_label = list(cfg.train_layers) if len(cfg.train_layers) > 0 else ["fc-only"]
        print(f"\nModel Config: {cfg.name}")
        print(f"  train_layers={layer_label}")
        print(f"  use_mixstyle={cfg.use_mixstyle} mixstyle_p={cfg.mixstyle_p} mixstyle_a={cfg.mixstyle_a}")
        print(f"  insert_after={insert_after}")
        print(f"  trainable_params={trainable_count}")
        print(f"  trainable_modules={trainable_names[:12]}{' ...' if len(trainable_names) > 12 else ''}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    return model, optimizer, criterion


# ---------------------------------------------------------------------------
# Experiment loops
# ---------------------------------------------------------------------------
def run_experiment(
    cfg: ModelConfig,
    data_cfg: DataSplitConfig,
    index_df: pd.DataFrame,
    epochs: int = 15,
    patience: int = config.EARLY_STOP_PATIENCE,
    seed: int = config.EXP_SEED,
    ckpt_fmt: str = "best_{name}.pth",
):
    """Train ``cfg`` on ``data_cfg`` with early stopping; evaluate on both test sets."""
    set_seed(seed)
    runtime.reset_scaler()
    loaders = prepare_dataloaders_from_config(data_cfg, index_df=index_df)
    model, optimizer, criterion = build_model(cfg)

    best_val = 0.0
    no_improve = 0
    history = []
    pth = ckpt_fmt.format(name=cfg.name)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, loaders["train_loader"], optimizer, criterion)
        val_acc = eval_epoch(model, loaders["val_loader"])
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"[{cfg.name}] Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), pth)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{cfg.name}] Early stopping at epoch {epoch+1} (patience={patience})")
                break

    model.load_state_dict(torch.load(pth))
    in_domain_test_acc = eval_epoch(model, loaders["test_loader"])
    out_domain_test_acc = eval_epoch(model, loaders["out_domain_test_loader"])

    return {
        "name": cfg.name,
        "best_val": best_val,
        "history": history,
        "ckpt": pth,
        "train_layers": list(cfg.train_layers),
        "mixstyle_p": cfg.mixstyle_p,
        "mixstyle_a": cfg.mixstyle_a,
        "data_config": data_cfg.name,
        "train_cases": sorted(data_cfg.train_cases),
        "validation_cases": sorted(data_cfg.validation_cases),
        "test_cases": sorted(data_cfg.test_cases),
        "in_domain_test_acc": in_domain_test_acc,
        "out_domain_test_acc": out_domain_test_acc,
    }


def run_loso_experiment(
    cfg: ModelConfig,
    held_out_site: int,
    index_df: pd.DataFrame,
    epochs: int = 5,
    patience: int = 3,
    seed: int = config.EXP_SEED,
):
    """Train one LOSO fold (site held out) and report in- vs out-domain metrics."""
    split_frames = make_loso_split(index_df, held_out_site=held_out_site)
    data_cfg = DataSplitConfig(
        name=f"loso-site-{held_out_site:02d}",
        train_cases=frozenset(split_frames["train_df"]["case_id"].unique().tolist()),
        validation_cases=frozenset(split_frames["val_df"]["case_id"].unique().tolist()),
        test_cases=frozenset(split_frames["in_domain_test_df"]["case_id"].unique().tolist()),
        out_domain_test_cases=frozenset(split_frames["out_domain_test_df"]["case_id"].unique().tolist()),
    )
    loaders = prepare_dataloaders_from_config(data_cfg, index_df=index_df)

    set_seed(seed + int(held_out_site))
    runtime.reset_scaler()
    model, optimizer, criterion = build_model(cfg)

    best_val = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, loaders["train_loader"], optimizer, criterion)
        val_acc = eval_epoch(model, loaders["val_loader"])
        print(
            f"[site {held_out_site:02d}] {cfg.name} epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[site {held_out_site:02d}] early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    in_metrics = eval_metrics(model, loaders["test_loader"])
    out_metrics = eval_metrics(model, loaders["out_domain_test_loader"])
    return {
        "held_out_site": held_out_site,
        "train_sites": sorted(split_frames["train_df"]["site"].unique().tolist()),
        "out_cases": split_frames["out_domain_test_df"]["case_id"].nunique(),
        "best_val": best_val,
        "in_acc": in_metrics["accuracy"],
        "in_bal_acc": in_metrics["balanced_accuracy"],
        "out_acc": out_metrics["accuracy"],
        "out_bal_acc": out_metrics["balanced_accuracy"],
        "generalization_gap": in_metrics["balanced_accuracy"] - out_metrics["balanced_accuracy"],
        "out_tumor_f1": out_metrics["tumor_f1"],
    }


def run_loso_sweep(cfg: ModelConfig, sites: list[int], index_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run ``run_loso_experiment`` across ``sites`` and return a ranked DataFrame."""
    print(f"Running LOSO evaluation with {cfg.name} across sites: {sites}")
    rows = [run_loso_experiment(cfg, site, index_df=index_df, **kwargs) for site in sites]
    return (
        pd.DataFrame(rows)
        .sort_values(["out_bal_acc", "generalization_gap"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def load_trained_model(cfg: ModelConfig, ckpt_fmt: str = "best_{name}.pth"):
    """Reload a trained model from its checkpoint."""
    model, _, _ = build_model(cfg, verbose=False)
    model.load_state_dict(torch.load(ckpt_fmt.format(name=cfg.name)))
    return model


def report_experiments(
    experiments: list[ModelConfig],
    loaders: dict,
    results_by_name: dict | None = None,
    ckpt_fmt: str = "best_{name}.pth",
):
    """Print in/out-domain metrics for each experiment checkpoint."""
    print("=== Report ===\n")
    records = []
    for cfg in experiments:
        model = load_trained_model(cfg, ckpt_fmt=ckpt_fmt)
        in_m = eval_metrics(model, loaders["test_loader"])
        out_m = eval_metrics(model, loaders["out_domain_test_loader"])
        best_val = results_by_name[cfg.name]["best_val"] if results_by_name and cfg.name in results_by_name else None
        records.append((cfg, best_val, in_m, out_m))
        val_str = f"{best_val:.4f}" if best_val is not None else "N/A"
        print(
            f"{cfg.name} | Layer: {cfg.train_layers} "
            f"MixStyle: {cfg.use_mixstyle} (p={cfg.mixstyle_p}, a={cfg.mixstyle_a}) | Val: {val_str}"
        )
        print(
            f"  In-domain Acc: {in_m['accuracy']:.4f} | Out-domain Acc: {out_m['accuracy']:.4f} | "
            f"In-domain Bal Acc: {in_m['balanced_accuracy']:.4f} | Out-domain Bal Acc: {out_m['balanced_accuracy']:.4f}"
        )
        print(
            f"  tumor P/R/F1 in-domain: "
            f"{in_m['tumor_precision']:.4f}/{in_m['tumor_recall']:.4f}/{in_m['tumor_f1']:.4f} | "
            f"out-domain: "
            f"{out_m['tumor_precision']:.4f}/{out_m['tumor_recall']:.4f}/{out_m['tumor_f1']:.4f}"
        )
    return records


def compare_models(model_cfgs: list[ModelConfig], loaders: dict, ckpt_fmt: str = "best_{name}.pth"):
    """Evaluate each config and, if exactly two are provided, print baseline-vs-compare deltas."""
    eval_results = []
    for cfg in model_cfgs:
        model = load_trained_model(cfg, ckpt_fmt=ckpt_fmt)
        in_m = eval_metrics(model, loaders["test_loader"])
        out_m = eval_metrics(model, loaders["out_domain_test_loader"])
        eval_results.append((cfg, in_m, out_m))
        print(f"{cfg.name}")
        print(
            f"  in-domain  acc={in_m['accuracy']:.4f} bal_acc={in_m['balanced_accuracy']:.4f} tumor_f1={in_m['tumor_f1']:.4f}"
        )
        print(
            f"  out-domain acc={out_m['accuracy']:.4f} bal_acc={out_m['balanced_accuracy']:.4f} tumor_f1={out_m['tumor_f1']:.4f}"
        )

    if len(eval_results) == 2:
        (b_cfg, b_in, b_out), (c_cfg, c_in, c_out) = eval_results
        print("\nComparison")
        print(f"baseline: {b_cfg.name}")
        print(f"compare:  {c_cfg.name}")
        print(f"delta in_acc:       {c_in['accuracy'] - b_in['accuracy']:+.4f}")
        print(f"delta in_bal_acc:   {c_in['balanced_accuracy'] - b_in['balanced_accuracy']:+.4f}")
        print(f"delta out_acc:      {c_out['accuracy'] - b_out['accuracy']:+.4f}")
        print(f"delta out_bal_acc:  {c_out['balanced_accuracy'] - b_out['balanced_accuracy']:+.4f}")
        print(f"delta out_tumor_f1: {c_out['tumor_f1'] - b_out['tumor_f1']:+.4f}")
        if b_out["accuracy"]:
            print(
                f"out_accuracy improvement: "
                f"{((c_out['accuracy'] - b_out['accuracy']) / b_out['accuracy'] * 100):+.2f}%"
            )

    return eval_results
