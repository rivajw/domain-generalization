#!/usr/bin/env python3
"""Reconstruct the paper's MixStyle + Multi study, on fixed BraTS LOSO splits.

This runner is separate from run_dg_multisite.py: it NEVER modifies or replaces
existing five-method experiment logs, model checkpoints, or all_runs.csv.
Defaults are a small smoke experiment (site18, seed0, strong MixStyle, w=1).
"""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
from pathlib import Path

import pandas as pd
from aws_paths import load_index, create_data_split, require_training_device
from run_dg_multisite import create_manifest, Tee

REGIMES = {'strong': (0.75, 0.1), 'weak': (0.5, 0.5)}
DEFAULT_SITES = (18, 1, 4, 13, 20)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def experiment_name(site, seed, split_seed, regime, w):
    p, alpha = REGIMES[regime]
    return (f'DG-mixstyle_multi-site{site}-L2-split{split_seed}-'
            f'seed{seed}-{regime}-p{p:g}-a{alpha:g}-w{w:g}')


def summarize(root, previous_results):
    records = [json.loads(path.read_text()) for path in sorted(root.glob('DG-mixstyle_multi-*_metrics.json'))]
    if not records:
        return
    frame = pd.DataFrame(records).sort_values(['site','regime','multi_weight','train_seed'])
    frame.to_csv(root / 'multi_all_runs.csv', index=False)
    keys = ['regime', 'multi_weight', 'site']
    grouped = frame.groupby(keys, as_index=False).agg(
        n=('ood_bal_acc','size'), in_mean=('in_bal_acc','mean'), in_std=('in_bal_acc','std'),
        ood_mean=('ood_bal_acc','mean'), ood_std=('ood_bal_acc','std'),
        f1_mean=('ood_tumor_f1','mean'), f1_std=('ood_tumor_f1','std'),
        gap_mean=('gap','mean'), gap_std=('gap','std'))
    grouped.to_csv(root / 'multi_by_site_summary.csv', index=False)
    grouped.groupby(['regime','multi_weight'], as_index=False).agg(
        sites=('site','size'), site_macro_in=('in_mean','mean'),
        site_macro_ood=('ood_mean','mean'), site_macro_f1=('f1_mean','mean'),
        site_macro_gap=('gap_mean','mean')).to_csv(root / 'multi_macro_summary.csv', index=False)

    # Optional convenience comparison. DO NOT overwrite the original all_runs.csv.
    if previous_results.is_file():
        old = pd.read_csv(previous_results)
        new = frame.copy()
        old['regime'] = None
        old['multi_weight'] = float('nan')
        pd.concat([old, new], ignore_index=True, sort=False).to_csv(root / 'with_existing_methods.csv', index=False)
    print('Saved multi-scale results:',root,flush=True)


def train_one(cfg, data_cfg, index_df, train_seed, epochs, patience, ckpt, weight):
    """Matches dg/experiment.py: Adam 1e-4, batch128, source val selection."""
    import torch
    from dg import runtime
    from dg.training import set_seed, eval_epoch
    from dg.experiment import build_model, prepare_dataloaders_from_config
    from dg.multiscale import train_epoch_multiscale
    set_seed(train_seed)
    runtime.reset_scaler()
    loaders = prepare_dataloaders_from_config(data_cfg, index_df, verbose=False)
    model, optimizer, criterion = build_model(cfg, verbose=True)
    best_val, patience_used, history = float('-inf'), 0, []
    for epoch in range(1, epochs + 1):
        metrics = train_epoch_multiscale(model, loaders['train_loader'], optimizer,
                                         criterion, weight=weight)
        val_acc = eval_epoch(model, loaders['val_loader'])
        history.append(dict(epoch=epoch, **metrics, val_acc=val_acc))
        print(f'[{cfg.name}] epoch {epoch}/{epochs}: '
              f'CE={metrics["ce"]:.4f} multi={metrics["multi"]:.4f} '
              f'train={metrics["train_acc"]:.4f} val={val_acc:.4f}',flush=True)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt)
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= patience:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break
    del model, optimizer, criterion, loaders
    return best_val, history


def evaluate(cfg, data_cfg, df, site, seed, split_seed, split_hash,
             regime, weight, best_val, ckpt):
    import torch
    from dg import runtime
    from dg.experiment import build_model, prepare_dataloaders_from_config
    from dg.training import eval_metrics
    loaders = prepare_dataloaders_from_config(data_cfg, df, verbose=False)
    model, _, _ = build_model(cfg, pretrained=False, verbose=False)
    model.load_state_dict(torch.load(ckpt, map_location=runtime.DEVICE, weights_only=True))
    model.eval()
    ind = eval_metrics(model, loaders['test_loader'])
    ood = eval_metrics(model, loaders['out_domain_test_loader'])
    del model, loaders
    return dict(method='mixstyle_multi', name=cfg.name, site=site,
                train_seed=seed, split_seed=split_seed, split_sha256=split_hash,
                regime=regime, multi_weight=weight,
                mixstyle_p=REGIMES[regime][0], mixstyle_alpha=REGIMES[regime][1],
                multi_reduction='sum_four_stage_cosine_flatten_per_sample',
                best_val=best_val,
                in_bal_acc=ind['balanced_accuracy'], ood_bal_acc=ood['balanced_accuracy'],
                in_acc=ind['accuracy'], ood_acc=ood['accuracy'],
                ood_tumor_f1=ood['tumor_f1'], ood_tumor_precision=ood['tumor_precision'],
                ood_tumor_recall=ood['tumor_recall'],
                gap=ind['balanced_accuracy']-ood['balanced_accuracy'],
                in_tp=ind['tp'], in_fp=ind['fp'], in_tn=ind['tn'], in_fn=ind['fn'],
                ood_tp=ood['tp'], ood_fp=ood['fp'], ood_tn=ood['tn'], ood_fn=ood['fn'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent / 'brats23_dg_png')
    ap.add_argument('--index', type=Path)
    ap.add_argument('--sites', nargs='+', type=int, default=[18])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--regimes', nargs='+', choices=REGIMES, default=['strong'])
    ap.add_argument('--weights', nargs='+', type=float, default=[1.0])
    ap.add_argument('--split-seed', type=int, default=18662)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow-cpu', action='store_true')
    args=ap.parse_args()
    if args.eval_only and args.force:
        ap.error('--eval-only and --force are mutually exclusive')
    if not args.sites or not args.seeds or not args.weights:
        ap.error('sites, seeds, weights cannot be empty')
    if any(w < 0 for w in args.weights):
        ap.error('All multi-scale weights must be >= 0')
    if len(set(args.sites))!=len(args.sites) or len(set(args.seeds))!=len(args.seeds):
        ap.error('duplicate sites/seeds not allowed')
    if len(set(args.weights))!=len(args.weights) or len(set(args.regimes))!=len(args.regimes):
        ap.error('duplicate weights/regimes not allowed')
    from dg import config as dgc
    from dg.experiment import ModelConfig
    index,_ = load_index(args.data_root, args.index)
    outdir = Path(dgc.RESULTS_DIR) / 'multiscale_runs'
    outdir.mkdir(parents=True, exist_ok=True)
    prior = Path(dgc.RESULTS_DIR) / 'multisite_runs' / 'all_runs.csv'
    logs = Path(dgc.ARTIFACTS_ROOT) / 'logs' / 'multiscale_runs'
    logs.mkdir(parents=True, exist_ok=True)
    Path(dgc.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print('Sites:',args.sites,'train seeds:',args.seeds,'split seed:',args.split_seed)
    print('Regimes:',args.regimes,'weights:',args.weights,'expected runs:',len(args.sites)*len(args.seeds)*len(args.regimes)*len(args.weights))
    if not args.dry_run and not args.eval_only:
        require_training_device(args.allow_cpu)
    for site in args.sites:
        data_cfg, split = create_data_split(index, site, args.split_seed)
        # The exact same manifest and SHA algorithm as earlier 125 runs.
        root_original = Path(dgc.RESULTS_DIR) / 'multisite_runs'
        root_original.mkdir(parents=True, exist_ok=True)
        checksum = create_manifest(index, split, site, args.split_seed, root_original)
        if prior.is_file():
            old = pd.read_csv(prior, usecols=['site','split_sha256'])
            checks = old.loc[old.site == site, 'split_sha256'].unique()
            if len(checks) and (len(checks)!=1 or checks[0] != checksum):
                raise RuntimeError(f'Data split for site {site} differs from the original 125-run study')
        if args.dry_run:
            continue
        for regime in args.regimes:
            p, alpha = REGIMES[regime]
            for w in args.weights:
                for seed in args.seeds:
                    name = experiment_name(site,seed,args.split_seed,regime,w)
                    cfg = ModelConfig(name=name, train_layers=(2,),use_mixstyle=True,
                                      mixstyle_p=p, mixstyle_a=alpha)
                    ckpt=Path(dgc.CHECKPOINT_DIR)/f'best_{name}.pth'
                    metrics_path=outdir/f'{name}_metrics.json'
                    history_path=outdir/f'{name}_history.json'
                    if metrics_path.is_file() and not args.force:
                        old=json.loads(metrics_path.read_text())
                        if old.get('split_sha256')!=checksum:
                            raise RuntimeError(f'Split checksum mismatch for {name}')
                        print('Skip complete:',name,flush=True)
                        continue
                    if args.eval_only and not ckpt.is_file():
                        print('Skip absent:',ckpt,flush=True)
                        continue
                    with (logs/f'{name}.log').open('a',buffering=1) as lf:
                        with contextlib.redirect_stdout(Tee(sys.stdout,lf)), contextlib.redirect_stderr(Tee(sys.stderr,lf)):
                            print('Training:',name,flush=True)
                            print('Split checksum:',checksum,flush=True)
                            if args.force or (not ckpt.is_file() and not args.eval_only):
                                best_val, history = train_one(cfg,data_cfg,index,seed,
                                                              args.epochs,args.patience,ckpt,w)
                                history_path.write_text(json.dumps({'name':name,'best_val':best_val,
                                         'history':history, 'multi_weight':w,
                                         'regime':regime,'split_sha256':checksum},indent=2))
                            else:
                                print('Using saved checkpoint:',ckpt,flush=True)
                            best_val = None
                            if history_path.is_file():
                                best_val=json.loads(history_path.read_text()).get('best_val')
                            row=evaluate(cfg,data_cfg,index,site,seed,args.split_seed,
                                         checksum,regime,w,best_val,ckpt)
                            metrics_path.write_text(json.dumps(row,indent=2))
                            print(pd.DataFrame([row])[["name","in_bal_acc","ood_bal_acc","ood_tumor_f1","gap"]].round(4).to_string(index=False),flush=True)
                    summarize(outdir,prior)
    if not args.dry_run:
        summarize(outdir,prior)

if __name__=='__main__':
    main()
