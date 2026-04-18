"""High-level experiment runners: configs, model builder, and LOSO + variant loops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import config, runtime
from .dataset import BraTSNPZSliceDataset
from .models import ResNet18MixStyle
from .splits import make_loso_split
from .training import (
    eval_epoch,
    eval_metrics,
    set_seed,
    train_epoch,
    train_epoch_consistency,
    train_epoch_mmd,
    train_epoch_consistency_mmd,
)


def _resolve_ckpt_path(ckpt_fmt: str, name: str) -> str:
    ckpt_dir = Path(config.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return str(ckpt_dir / ckpt_fmt.format(name=name))


def _resolve_result_path(name: str) -> Path:
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{name}.json"


def _save_experiment_result(result: dict) -> None:
    result_path = _resolve_result_path(result["name"])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_experiment_result(name: str) -> dict | None:
    result_path = _resolve_result_path(name)
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


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
class StageConfig:
    # Optional target overrides; when None, fall back to ModelConfig static values.
    consistency_lambda_target: Optional[float] = None
    consistency_warmup_epochs: int = 0
    consistency_ramp_epochs: int = 0
    consistency_start_value: float = 0.0

    mmd_lambda_target: Optional[float] = None
    mmd_warmup_epochs: int = 0
    mmd_ramp_epochs: int = 0
    mmd_start_value: float = 0.0

    mixstyle_enable_after_epoch: int = 0
    mixstyle_p_start: Optional[float] = None
    mixstyle_p_target: Optional[float] = None
    mixstyle_p_ramp_epochs: int = 0

    mixstyle_a_start: Optional[float] = None
    mixstyle_a_target: Optional[float] = None
    mixstyle_a_ramp_epochs: int = 0


def _linear_warmup_ramp(
    epoch: int,
    target_value: float,
    warmup_epochs: int = 0,
    ramp_epochs: int = 0,
    start_value: float = 0.0,
) -> float:
    if epoch < warmup_epochs:
        return start_value
    if ramp_epochs <= 0:
        return target_value

    t = epoch - warmup_epochs
    if t >= ramp_epochs:
        return target_value

    ratio = t / ramp_epochs
    return start_value + ratio * (target_value - start_value)


def _resolve_epoch_stage_values(cfg: "ModelConfig", epoch: int) -> dict:
    if cfg.stage_cfg is None:
        return {
            "lambda_c": float(cfg.consistency_lambda),
            "lambda_mmd": float(cfg.mmd_lambda),
            "mixstyle_enabled": bool(cfg.use_mixstyle),
            "mixstyle_p": float(cfg.mixstyle_p),
            "mixstyle_a": float(cfg.mixstyle_a),
        }

    stage_cfg = cfg.stage_cfg

    consistency_target = (
        cfg.consistency_lambda
        if stage_cfg.consistency_lambda_target is None
        else stage_cfg.consistency_lambda_target
    )
    mmd_target = cfg.mmd_lambda if stage_cfg.mmd_lambda_target is None else stage_cfg.mmd_lambda_target

    lambda_c = _linear_warmup_ramp(
        epoch=epoch,
        target_value=consistency_target,
        warmup_epochs=stage_cfg.consistency_warmup_epochs,
        ramp_epochs=stage_cfg.consistency_ramp_epochs,
        start_value=stage_cfg.consistency_start_value,
    )
    lambda_mmd = _linear_warmup_ramp(
        epoch=epoch,
        target_value=mmd_target,
        warmup_epochs=stage_cfg.mmd_warmup_epochs,
        ramp_epochs=stage_cfg.mmd_ramp_epochs,
        start_value=stage_cfg.mmd_start_value,
    )

    mixstyle_enabled = cfg.use_mixstyle and (epoch >= stage_cfg.mixstyle_enable_after_epoch)

    mixstyle_p_target = cfg.mixstyle_p if stage_cfg.mixstyle_p_target is None else stage_cfg.mixstyle_p_target
    mixstyle_a_target = cfg.mixstyle_a if stage_cfg.mixstyle_a_target is None else stage_cfg.mixstyle_a_target
    mixstyle_p_start = mixstyle_p_target if stage_cfg.mixstyle_p_start is None else stage_cfg.mixstyle_p_start
    mixstyle_a_start = mixstyle_a_target if stage_cfg.mixstyle_a_start is None else stage_cfg.mixstyle_a_start

    rel_epoch = max(0, epoch - stage_cfg.mixstyle_enable_after_epoch)
    mixstyle_p = _linear_warmup_ramp(
        epoch=rel_epoch,
        target_value=mixstyle_p_target,
        warmup_epochs=0,
        ramp_epochs=stage_cfg.mixstyle_p_ramp_epochs,
        start_value=mixstyle_p_start,
    )
    mixstyle_a = _linear_warmup_ramp(
        epoch=rel_epoch,
        target_value=mixstyle_a_target,
        warmup_epochs=0,
        ramp_epochs=stage_cfg.mixstyle_a_ramp_epochs,
        start_value=mixstyle_a_start,
    )

    if not mixstyle_enabled:
        mixstyle_p = 0.0

    return {
        "lambda_c": float(lambda_c),
        "lambda_mmd": float(lambda_mmd),
        "mixstyle_enabled": bool(mixstyle_enabled),
        "mixstyle_p": float(mixstyle_p),
        "mixstyle_a": float(mixstyle_a),
    }


@dataclass(frozen=True)
class ModelConfig:
    name: str
    train_layers: tuple[int, ...] = tuple()
    use_mixstyle: bool = False
    mixstyle_p: float = 0.5
    mixstyle_a: float = 0.1

    # KL consistency regularization
    consistency_lambda: float = 0.0

    # MMD regularization
    mmd_lambda: float = 0.0
    mmd_class_conditional: bool = True
    mmd_sigma_list: tuple[float, ...] = (1.0, 5.0, 10.0)
    mmd_max_samples: int | None = 256
    mmd_on_clean_features: bool = False

    # Optional staged schedule overrides.
    stage_cfg: Optional[StageConfig] = None


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
        if cfg.stage_cfg is not None:
            print(f"  stage_cfg={cfg.stage_cfg}")
        print(f"  insert_after={insert_after}")
        print(f"  trainable_params={trainable_count}")
        print(f"  trainable_modules={trainable_names[:12]}{' ...' if len(trainable_names) > 12 else ''}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    return model, optimizer, criterion


# ---------------------------------------------------------------------------
# Experiment loops
# ---------------------------------------------------------------------------
def _uses_consistency(cfg: ModelConfig) -> bool:
    if not cfg.use_mixstyle:
        return False
    if cfg.stage_cfg is not None and cfg.stage_cfg.consistency_lambda_target is not None:
        return cfg.stage_cfg.consistency_lambda_target > 0
    return cfg.consistency_lambda > 0


def _uses_mmd(cfg: ModelConfig) -> bool:
    if cfg.stage_cfg is not None and cfg.stage_cfg.mmd_lambda_target is not None:
        return cfg.stage_cfg.mmd_lambda_target > 0
    return cfg.mmd_lambda > 0


def _train_one_epoch(cfg: ModelConfig, model, loaders, optimizer, criterion, epoch: int):
    stage_vals = _resolve_epoch_stage_values(cfg, epoch)
    use_consistency = _uses_consistency(cfg)
    use_mmd = _uses_mmd(cfg)

    if use_consistency and use_mmd:
        train_loss, ce_loss, reg1_loss, reg2_loss, train_acc = train_epoch_consistency_mmd(
            model,
            loaders["train_loader"],
            optimizer,
            criterion,
            lambda_c=stage_vals["lambda_c"],
            lambda_mmd=stage_vals["lambda_mmd"],
            class_conditional=cfg.mmd_class_conditional,
            sigma_list=cfg.mmd_sigma_list,
            max_samples=cfg.mmd_max_samples,
            mixstyle_enabled=stage_vals["mixstyle_enabled"],
            mixstyle_p=stage_vals["mixstyle_p"],
            mixstyle_a=stage_vals["mixstyle_a"],
            mmd_on_clean_features=cfg.mmd_on_clean_features,
        )
        log = {
            "train_loss": train_loss,
            "ce_loss": ce_loss,
            "kl_loss": reg1_loss,
            "mmd_loss": reg2_loss,
            "train_acc": train_acc,
            "lambda_c": stage_vals["lambda_c"],
            "lambda_mmd": stage_vals["lambda_mmd"],
            "mixstyle_enabled": stage_vals["mixstyle_enabled"],
            "mixstyle_p": stage_vals["mixstyle_p"],
            "mixstyle_a": stage_vals["mixstyle_a"],
        }
        extra = (
            f" ce={ce_loss:.4f} kl={reg1_loss:.4f} mmd={reg2_loss:.4f}"
            f" lc={stage_vals['lambda_c']:.4f} lm={stage_vals['lambda_mmd']:.4f}"
            f" ms={int(stage_vals['mixstyle_enabled'])} p={stage_vals['mixstyle_p']:.3f} a={stage_vals['mixstyle_a']:.3f}"
        )
        return log, extra

    if use_consistency:
        train_loss, ce_loss, kl_loss, train_acc = train_epoch_consistency(
            model,
            loaders["train_loader"],
            optimizer,
            criterion,
            lambda_c=stage_vals["lambda_c"],
            mixstyle_enabled=stage_vals["mixstyle_enabled"],
            mixstyle_p=stage_vals["mixstyle_p"],
            mixstyle_a=stage_vals["mixstyle_a"],
        )
        log = {
            "train_loss": train_loss,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "train_acc": train_acc,
            "lambda_c": stage_vals["lambda_c"],
            "lambda_mmd": 0.0,
            "mixstyle_enabled": stage_vals["mixstyle_enabled"],
            "mixstyle_p": stage_vals["mixstyle_p"],
            "mixstyle_a": stage_vals["mixstyle_a"],
        }
        extra = (
            f" ce={ce_loss:.4f} kl={kl_loss:.4f}"
            f" lc={stage_vals['lambda_c']:.4f}"
            f" ms={int(stage_vals['mixstyle_enabled'])} p={stage_vals['mixstyle_p']:.3f} a={stage_vals['mixstyle_a']:.3f}"
        )
        return log, extra

    if use_mmd:
        train_loss, ce_loss, mmd_loss, train_acc = train_epoch_mmd(
            model,
            loaders["train_loader"],
            optimizer,
            criterion,
            lambda_mmd=stage_vals["lambda_mmd"],
            class_conditional=cfg.mmd_class_conditional,
            sigma_list=cfg.mmd_sigma_list,
            max_samples=cfg.mmd_max_samples,
            mixstyle_enabled=stage_vals["mixstyle_enabled"],
            mixstyle_p=stage_vals["mixstyle_p"],
            mixstyle_a=stage_vals["mixstyle_a"],
        )
        log = {
            "train_loss": train_loss,
            "ce_loss": ce_loss,
            "mmd_loss": mmd_loss,
            "train_acc": train_acc,
            "lambda_c": 0.0,
            "lambda_mmd": stage_vals["lambda_mmd"],
            "mixstyle_enabled": stage_vals["mixstyle_enabled"],
            "mixstyle_p": stage_vals["mixstyle_p"],
            "mixstyle_a": stage_vals["mixstyle_a"],
        }
        extra = (
            f" ce={ce_loss:.4f} mmd={mmd_loss:.4f}"
            f" lm={stage_vals['lambda_mmd']:.4f}"
            f" ms={int(stage_vals['mixstyle_enabled'])} p={stage_vals['mixstyle_p']:.3f} a={stage_vals['mixstyle_a']:.3f}"
        )
        return log, extra

    train_loss, train_acc = train_epoch(model, loaders["train_loader"], optimizer, criterion)
    log = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "lambda_c": 0.0,
        "lambda_mmd": 0.0,
        "mixstyle_enabled": stage_vals["mixstyle_enabled"],
        "mixstyle_p": stage_vals["mixstyle_p"],
        "mixstyle_a": stage_vals["mixstyle_a"],
    }
    return log, ""


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
    pth = _resolve_ckpt_path(ckpt_fmt, cfg.name)

    for epoch in range(epochs):
        epoch_log, extra = _train_one_epoch(cfg, model, loaders, optimizer, criterion, epoch)
        epoch_log["epoch"] = epoch + 1
        epoch_log["val_acc"] = None
        history.append(epoch_log)

        train_loss = epoch_log["train_loss"]
        train_acc = epoch_log["train_acc"]
        val_acc = eval_epoch(model, loaders["val_loader"])
        history[-1]["val_acc"] = val_acc
        print(
            f"[{cfg.name}] Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f}{extra} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
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

    model.load_state_dict(torch.load(pth, map_location=runtime.DEVICE))
    in_domain_test_acc = eval_epoch(model, loaders["test_loader"])
    out_domain_test_acc = eval_epoch(model, loaders["out_domain_test_loader"])

    result = {
        "name": cfg.name,
        "best_val": best_val,
        "history": history,
        "ckpt": pth,
        "train_layers": list(cfg.train_layers),
        "mixstyle_p": cfg.mixstyle_p,
        "mixstyle_a": cfg.mixstyle_a,
        "consistency_lambda": cfg.consistency_lambda,
        "mmd_lambda": cfg.mmd_lambda,
        "mmd_class_conditional": cfg.mmd_class_conditional,
        "mmd_on_clean_features": cfg.mmd_on_clean_features,
        "stage_cfg": None if cfg.stage_cfg is None else cfg.stage_cfg.__dict__,
        "data_config": data_cfg.name,
        "train_cases": sorted(data_cfg.train_cases),
        "validation_cases": sorted(data_cfg.validation_cases),
        "test_cases": sorted(data_cfg.test_cases),
        "in_domain_test_acc": in_domain_test_acc,
        "out_domain_test_acc": out_domain_test_acc,
    }
    _save_experiment_result(result)
    return result


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
        epoch_log, extra = _train_one_epoch(cfg, model, loaders, optimizer, criterion, epoch)
        train_loss = epoch_log["train_loss"]
        train_acc = epoch_log["train_acc"]
        val_acc = eval_epoch(model, loaders["val_loader"])
        print(
            f"[site {held_out_site:02d}] {cfg.name} epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f}{extra} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
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
    ckpt_path = _resolve_ckpt_path(ckpt_fmt, cfg.name)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {cfg.name}: {ckpt_path}. "
            "Run the training cell for this config first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=runtime.DEVICE))
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
        try:
            model = load_trained_model(cfg, ckpt_fmt=ckpt_fmt)
        except FileNotFoundError as exc:
            print(f"{cfg.name} | checkpoint missing")
            print(f"  {exc}")
            continue
        in_m = eval_metrics(model, loaders["test_loader"])
        out_m = eval_metrics(model, loaders["out_domain_test_loader"])
        saved_result = load_experiment_result(cfg.name)
        best_val = None
        if results_by_name and cfg.name in results_by_name:
            best_val = results_by_name[cfg.name]["best_val"]
        elif saved_result is not None:
            best_val = saved_result.get("best_val")
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
        try:
            model = load_trained_model(cfg, ckpt_fmt=ckpt_fmt)
        except FileNotFoundError as exc:
            print(f"{cfg.name}")
            print(f"  {exc}")
            continue
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
