#!/usr/bin/env python3
"""AWS CLI conversion of final_report.ipynb (original MixStyle/KL/MMD workflow).

No Colab mounts or Jupyter magics. Uses original dg training/evaluation modules.
Training and diagnostic stages can be launched independently; see --help.
The supplied dg/ trainer does NOT support simultaneous KL+MMD, so no new
combined-loss experiment is fabricated here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aws_paths import add_common_args, create_data_split, load_index, require_training_device, save_split_manifest


def make_configs():
    from dg.experiment import ModelConfig
    original = [
        ModelConfig(name="Baseline-L1", train_layers=(1,), use_mixstyle=False),
        ModelConfig(name="Baseline-L2", train_layers=(2,), use_mixstyle=False),
        ModelConfig(name="Baseline-L3", train_layers=(3,), use_mixstyle=False),
        ModelConfig(name="MixStyle-L1-p0.5-a0.1", train_layers=(1,), use_mixstyle=True, mixstyle_p=0.5, mixstyle_a=0.1),
        ModelConfig(name="MixStyle-L2-p0.5-a0.1", train_layers=(2,), use_mixstyle=True, mixstyle_p=0.5, mixstyle_a=0.1),
        ModelConfig(name="MixStyle-L3-p0.5-a0.1", train_layers=(3,), use_mixstyle=True, mixstyle_p=0.5, mixstyle_a=0.1),
        ModelConfig(name="MixStyle-L2-p0.25-a0.1", train_layers=(2,), use_mixstyle=True, mixstyle_p=0.25, mixstyle_a=0.1),
        ModelConfig(name="MixStyle-L2-p0.75-a0.1", train_layers=(2,), use_mixstyle=True, mixstyle_p=0.75, mixstyle_a=0.1),
        ModelConfig(name="MixStyle-L2-p0.5-a0.3", train_layers=(2,), use_mixstyle=True, mixstyle_p=0.5, mixstyle_a=0.3),
        ModelConfig(name="MixStyle-L2-p0.5-a0.5", train_layers=(2,), use_mixstyle=True, mixstyle_p=0.5, mixstyle_a=0.5),
    ]
    consistency = [
        ModelConfig(name=f"ConsistMix-L2-lam{lam}", train_layers=(2,), use_mixstyle=True,
                    mixstyle_p=0.75, mixstyle_a=0.1, consistency_lambda=lam)
        for lam in (0.05, 0.1, 0.5, 1.0)
    ]
    mmd = [
        ModelConfig(name=f"MixMMD-L2-lam{lam}", train_layers=(2,), use_mixstyle=True,
                    mixstyle_p=0.75, mixstyle_a=0.1, mmd_lambda=lam)
        for lam in (0.05, 0.1, 0.5)
    ]
    optimal = [ModelConfig(name="MixStyle-L3-p0.75-a0.5", train_layers=(3,),
                           use_mixstyle=True, mixstyle_p=0.75, mixstyle_a=0.5)]
    return {
        "baseline": original[:3], "mixstyle": original[3:],
        "kl": consistency, "mmd": mmd, "optimal": optimal,
    }


def selected_configs(groups, group):
    if group == "all":
        return sum(groups.values(), [])
    return groups[group]


def train_models(cfgs, data_cfg, index_df, args):
    from dg.experiment import run_experiment, load_experiment_result
    from dg import config
    for n, cfg in enumerate(cfgs, 1):
        cp = Path(config.CHECKPOINT_DIR) / f"best_{cfg.name}.pth"
        if cp.exists() and not args.force:
            print(f"[{n}/{len(cfgs)}] SKIP existing checkpoint {cp} (use --force to retrain)", flush=True)
            continue
        print(f"[{n}/{len(cfgs)}] TRAIN {cfg.name}", flush=True)
        output = run_experiment(cfg, data_cfg, index_df=index_df,
                                epochs=args.epochs, patience=args.patience, seed=args.seed)
        print(f"Saved checkpoint={output['ckpt']}, source val={output['best_val']:.4f}", flush=True)


def evaluate_models(cfgs, data_cfg, index_df, args):
    from dg.experiment import load_trained_model, prepare_dataloaders_from_config, load_experiment_result
    from dg.training import eval_metrics
    from dg import config

    loaders = prepare_dataloaders_from_config(data_cfg, index_df=index_df, verbose=False)
    rows = []
    for cfg in cfgs:
        ckpt = Path(config.CHECKPOINT_DIR) / f"best_{cfg.name}.pth"
        if not ckpt.exists():
            print(f"SKIP missing checkpoint: {cfg.name}", flush=True)
            continue
        model = load_trained_model(cfg)
        in_m = eval_metrics(model, loaders["test_loader"])
        out_m = eval_metrics(model, loaders["out_domain_test_loader"])
        saved = load_experiment_result(cfg.name)
        rows.append({
            "model": cfg.name, "held_out_site": args.site,
            "source_val_acc": saved.get("best_val") if saved else None,
            "in_acc": in_m["accuracy"], "in_bal_acc": in_m["balanced_accuracy"],
            "ood_acc": out_m["accuracy"], "ood_bal_acc": out_m["balanced_accuracy"],
            "ood_tumor_precision": out_m["tumor_precision"],
            "ood_tumor_recall": out_m["tumor_recall"],
            "ood_tumor_f1": out_m["tumor_f1"],
            "gap": in_m["balanced_accuracy"] - out_m["balanced_accuracy"],
        })
        print(f"{cfg.name}: OOD BalAcc={out_m['balanced_accuracy']:.4f}, F1={out_m['tumor_f1']:.4f}", flush=True)
        del model
    table = pd.DataFrame(rows)
    dest = Path(config.RESULTS_DIR) / f"original_site{args.site}_{args.group}_comparison.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest, index=False)
    print(f"Saved results: {dest}", flush=True)
    return table


def measure_consistency_models(data_cfg, index_df, args):
    from dg.experiment import ModelConfig, prepare_dataloaders_from_config, load_trained_model
    from dg.consistency import measure_consistency, consistency_report
    from dg import config
    loaders = prepare_dataloaders_from_config(data_cfg, index_df=index_df, verbose=False)
    variants = [
        ModelConfig(name="MixStyle-L2-p0.75-a0.1", train_layers=(2,), use_mixstyle=True,
                    mixstyle_p=0.75, mixstyle_a=0.1),
        ModelConfig(name="MixStyle-L2-p0.5-a0.5", train_layers=(2,), use_mixstyle=True,
                    mixstyle_p=0.5, mixstyle_a=0.5),
        # Original notebook chose lam0.1 for this diagnostic by index, not best by OOD.
        ModelConfig(name="ConsistMix-L2-lam0.1", train_layers=(2,), use_mixstyle=True,
                    mixstyle_p=0.75, mixstyle_a=0.1, consistency_lambda=0.1),
    ]
    for cfg in variants:
        cp = Path(config.CHECKPOINT_DIR) / f"best_{cfg.name}.pth"
        if not cp.exists():
            print("SKIP missing checkpoint:", cp, flush=True)
            continue
        model = load_trained_model(cfg)
        for label, loader_key in (("train", "train_loader"), ("OOD", "out_domain_test_loader")):
            print(f"\n=== {cfg.name} {label} consistency ===", flush=True)
            records = measure_consistency(model, loaders[loader_key], n_repeats=args.repeats)
            consistency_report(records)


def measure_mmd_model(data_cfg, index_df, args):
    from dg.experiment import ModelConfig, prepare_dataloaders_from_config, load_trained_model
    from dg.mmd import measure_mmd_alignment, mmd_alignment_report
    from dg import config
    cfg = ModelConfig(name="MixMMD-L2-lam0.5", train_layers=(2,), use_mixstyle=True,
                      mixstyle_p=0.75, mixstyle_a=0.1, mmd_lambda=0.5)
    cp = Path(config.CHECKPOINT_DIR) / f"best_{cfg.name}.pth"
    if not cp.exists():
        print("SKIP missing checkpoint:", cp, flush=True)
        return
    loaders = prepare_dataloaders_from_config(data_cfg, index_df=index_df, verbose=False)
    model = load_trained_model(cfg)
    records = measure_mmd_alignment(model, loaders["train_loader"], n_repeats=args.repeats,
                                    class_conditional=True)
    mmd_alignment_report(records, title="TRAIN MMD ALIGNMENT")


def do_loso(index_df, args):
    from dg.experiment import ModelConfig, run_loso_sweep
    from dg.splits import build_site_domain_summary
    from dg import config
    summary = build_site_domain_summary(index_df)
    candidates = summary.loc[summary["candidate_for_loso"], "site"].tolist()
    print("LOSO candidate sites:", candidates, flush=True)
    sweep = run_loso_sweep(ModelConfig(name="LOSO-FC-Only", train_layers=tuple(),
                                     use_mixstyle=False), sites=candidates,
                           index_df=index_df, epochs=args.loso_epochs,
                           patience=args.patience, seed=args.seed)
    p = Path(config.RESULTS_DIR) / "loso_fc_only.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(p, index=False)
    print(sweep.to_string(index=False), "\nSaved:", p, flush=True)


def do_preprocess(args):
    from dg.preprocessing import preprocess_zip_to_npz
    from dg import config
    if args.raw_zip is None or args.mapping_xlsx is None:
        raise ValueError("--stage preprocess requires --raw-zip and --mapping-xlsx")
    if not args.raw_zip.is_file() or not args.mapping_xlsx.is_file():
        raise FileNotFoundError("BraTS raw ZIP or mapping spreadsheet not found")
    print("Running the same original preprocessing method (this can take substantial disk space).", flush=True)
    df = preprocess_zip_to_npz(zip_path=str(args.raw_zip), mapping_xlsx=str(args.mapping_xlsx),
                               out_root=str(args.data_root.resolve()))
    print("Preprocessed cases:", len(df), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--stage", choices=["data", "preprocess", "loso", "train", "report",
                                            "consistency", "mmd-analysis", "all"], default="data")
    parser.add_argument("--group", choices=["baseline", "mixstyle", "kl", "mmd", "optimal", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite matching old checkpoints")
    parser.add_argument("--repeats", type=int, default=5, help="Stochastic passes in diagnostics")
    parser.add_argument("--loso-epochs", type=int, default=5)
    parser.add_argument("--raw-zip", type=Path, default=None, help="Raw BraTS archive for preprocessing only")
    parser.add_argument("--mapping-xlsx", type=Path, default=None, help="Site mapping sheet for preprocessing only")
    args = parser.parse_args()

    if args.stage == "preprocess":
        do_preprocess(args)
        return

    from dg.training import set_seed
    set_seed(args.seed)
    df, index_path = load_index(args.data_root, args.index)
    cfg, split = create_data_split(df, args.site, args.seed)
    save_split_manifest(split, args.site, args.seed)
    if args.dry_run or args.stage == "data":
        return
    if args.stage in ("loso", "train", "all"):
        require_training_device(args.allow_cpu)
    groups = make_configs()
    configs = selected_configs(groups, args.group)
    if args.stage == "loso":
        do_loso(df, args)
    elif args.stage == "train":
        train_models(configs, cfg, df, args)
    elif args.stage == "report":
        evaluate_models(configs, cfg, df, args)
    elif args.stage == "consistency":
        measure_consistency_models(cfg, df, args)
    elif args.stage == "mmd-analysis":
        measure_mmd_model(cfg, df, args)
    elif args.stage == "all":
        # LOSO and preprocessing are separate costly stages, run manually if needed.
        train_models(configs, cfg, df, args)
        evaluate_models(configs, cfg, df, args)
        measure_consistency_models(cfg, df, args)
        measure_mmd_model(cfg, df, args)


if __name__ == "__main__":
    main()
