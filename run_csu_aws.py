#!/usr/bin/env python3
"""AWS CLI conversion of CSU_BraTS_experiments.ipynb.

CSU-1 default: site18, ResNet18, layer2, p=.75, alpha=.1, CE-only,
seed18662, max15 epochs, early-stop patience3. Never preprocesses the data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from aws_paths import add_common_args, create_data_split, load_index, require_training_device, save_split_manifest


def csu_config(seed, p, alpha, layer=2, site=18):
    from dg.experiment import ModelConfig
    # Preserve existing site-18 checkpoint names for backward compatibility.
    # Distinguish other sites so experiments never share checkpoint paths.
    name = f"CSU-L{layer}-p{p:g}-a{alpha:g}-seed{seed}"
    if site != 18:
        name += f"-site{site}"
    return ModelConfig(
        name=name,
        train_layers=(layer,), use_mixstyle=False, use_csu=True,
        csu_p=p, csu_alpha=alpha,
        consistency_lambda=0.0, mmd_lambda=0.0,
    )


def smoke_test(seed):
    import torch
    from dg import runtime
    from dg.csu import CSU
    from dg.training import set_seed
    set_seed(seed)
    module = CSU(p=1.0, alpha=0.1).to(runtime.DEVICE)
    features = torch.randn(8, 128, 8, 8, device=runtime.DEVICE, requires_grad=True)
    out = module(features)
    assert out.shape == features.shape and torch.isfinite(out).all()
    out.mean().backward()
    assert torch.isfinite(features.grad).all()
    module.eval()
    assert torch.equal(module(features.detach()), features.detach())
    print("CSU smoke test passed (shape, finite, backward, inference toggle)", flush=True)


def run_or_load(model_cfg, data_cfg, df, args, seed):
    from dg.experiment import run_experiment, load_experiment_result
    from dg import config
    cp = Path(config.CHECKPOINT_DIR) / f"best_{model_cfg.name}.pth"
    if args.eval_only or (cp.exists() and not args.force):
        if not cp.exists():
            raise FileNotFoundError(f"--eval-only requested but checkpoint missing: {cp}")
        print(f"Using existing checkpoint: {cp}", flush=True)
        return load_experiment_result(model_cfg.name)
    require_training_device(args.allow_cpu)
    return run_experiment(model_cfg, data_cfg, index_df=df, epochs=args.epochs,
                          patience=args.patience, seed=seed)


def metrics_row(model_cfg, loaders, site, seed, result):
    from dg.experiment import load_trained_model
    from dg.training import eval_metrics
    model = load_trained_model(model_cfg)
    id_m = eval_metrics(model, loaders["test_loader"])
    ood_m = eval_metrics(model, loaders["out_domain_test_loader"])
    return {
        "method": "CSU", "name": model_cfg.name, "held_out_site": site,
        "seed": seed, "csu_p": model_cfg.csu_p, "csu_alpha": model_cfg.csu_alpha,
        "val_acc": result.get("best_val") if result else None,
        "in_acc": id_m["accuracy"], "in_bal_acc": id_m["balanced_accuracy"],
        "ood_acc": ood_m["accuracy"], "ood_bal_acc": ood_m["balanced_accuracy"],
        "tumor_p": ood_m["tumor_precision"], "tumor_r": ood_m["tumor_recall"],
        "tumor_f1": ood_m["tumor_f1"],
        "gap": id_m["balanced_accuracy"] - ood_m["balanced_accuracy"],
    }


def comparison(loaders, cfg):
    from dg.experiment import ModelConfig, load_trained_model
    from dg.training import eval_metrics
    from dg import config
    layer = cfg.train_layers[0]
    references = [
        ModelConfig(name=f"Baseline-L{layer}", train_layers=(layer,), use_mixstyle=False),
        ModelConfig(name=f"MixStyle-L{layer}-p0.75-a0.1", train_layers=(layer,),
                    use_mixstyle=True, mixstyle_p=0.75, mixstyle_a=0.1),
        cfg,
    ]
    rows = []
    for ref in references:
        if not (Path(config.CHECKPOINT_DIR) / f"best_{ref.name}.pth").exists():
            print("Skipping missing checkpoint:", ref.name, flush=True)
            continue
        m = load_trained_model(ref)
        inc = eval_metrics(m, loaders["test_loader"])
        out = eval_metrics(m, loaders["out_domain_test_loader"])
        rows.append({"model": ref.name, "in_bal_acc": inc["balanced_accuracy"],
                     "ood_bal_acc": out["balanced_accuracy"],
                     "ood_tumor_f1": out["tumor_f1"],
                     "gap": inc["balanced_accuracy"] - out["balanced_accuracy"]})
        del m
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--p", type=float, default=0.75, help="CSU batch-level activation probability")
    parser.add_argument("--alpha", type=float, default=0.1, help="CSU perturbation-amplitude Beta parameter")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3], default=2,
                        help="ResNet stage to fine-tune and augment (L1/L2/L3)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate saved checkpoint, do not train")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--multiseed", action="store_true", help="Train/evaluate seeds 18662,18663,18664")
    parser.add_argument("--no-compare", action="store_true", help="Skip optional existing-checkpoint comparison")
    parser.add_argument("--no-smoke-test", action="store_true", help="Skip CSU numerical smoke test")
    args = parser.parse_args()
    if args.force and args.eval_only:
        parser.error("--force and --eval-only cannot both be specified")
    if not 0 <= args.p <= 1:
        parser.error("--p must be between 0 and 1")
    if args.alpha <= 0:
        parser.error("--alpha must be greater than 0")

    from dg import config, runtime
    from dg.training import set_seed
    set_seed(args.seed)
    df, _ = load_index(args.data_root, args.index)
    data_cfg, split = create_data_split(df, args.site, args.seed)
    save_split_manifest(split, args.site, args.seed)
    if args.dry_run:
        return

    if not args.no_smoke_test:
        smoke_test(args.seed)
    from dg.experiment import prepare_dataloaders_from_config
    loaders = None  # Created after training to avoid retaining duplicate in-memory NPZ datasets.
    seeds = [18662, 18663, 18664] if args.multiseed else [args.seed]
    rows = []
    for seed in seeds:
        cfg = csu_config(seed, args.p, args.alpha, args.layer, args.site)
        print("CSU experiment:", cfg, flush=True)
        result = run_or_load(cfg, data_cfg, df, args, seed)
        if loaders is None:
            loaders = prepare_dataloaders_from_config(data_cfg, index_df=df, verbose=False)
        row = metrics_row(cfg, loaders, args.site, seed, result)
        rows.append(row)
        print(pd.DataFrame([row]).round(4).to_string(index=False), flush=True)

    dst = Path(config.RESULTS_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    tag = f"CSU-site{args.site}-L{args.layer}-p{args.p:g}-a{args.alpha:g}"
    if args.multiseed:
        tag += "-seeds18662-18664"
    else:
        tag += f"-seed{args.seed}"
    output = dst / f"{tag}_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print("CSU results:", output, flush=True)

    if args.multiseed:
        print("Three-seed mean and std:")
        print(pd.DataFrame(rows)[["ood_bal_acc", "tumor_f1", "gap"]].agg(["mean", "std"]).to_string(), flush=True)
    elif not args.no_compare:
        table = comparison(loaders, csu_config(args.seed, args.p, args.alpha, args.layer, args.site))
        cmp_file = dst / f"{tag}_vs_existing.csv"
        table.to_csv(cmp_file, index=False)
        print(table.round(4).to_string(index=False), "\nSaved:", cmp_file, flush=True)


if __name__ == "__main__":
    main()
