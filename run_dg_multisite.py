#!/usr/bin/env python3
"""Fixed-split, five-seed / five-site BraTS DG study.

NEW experiment protocol. Site 18 old scores are NOT automatically combined with
these experiments; all methods receive identical data split + training seeds.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from aws_paths import load_index, create_data_split, require_training_device

DEFAULT_SITES = (18, 1, 4, 13, 20)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
METHODS = ('erm', 'mixstyle', 'csu', 'mixstyle_kl', 'mixstyle_kl_mmd')


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for f in self.streams:
            f.write(msg)
        return len(msg)
    def flush(self):
        for f in self.streams:
            f.flush()


def config_for(method, site, train_seed, split_seed, csu_p, csu_alpha):
    from dg.experiment import ModelConfig
    name = (f'DG-{method}-site{site}-L2-split{split_seed}-'
            f'seed{train_seed}')
    if method == 'erm':
        return ModelConfig(name=name, train_layers=(2,))
    if method == 'csu':
        name += f'-p{csu_p:g}-a{csu_alpha:g}'
        return ModelConfig(name=name, train_layers=(2,),
                           use_csu=True, csu_p=csu_p, csu_alpha=csu_alpha)
    params = dict(name=name, train_layers=(2,), use_mixstyle=True,
                  mixstyle_p=0.75, mixstyle_a=0.1)
    if method == 'mixstyle_kl':
        params['consistency_lambda'] = 1.0
    elif method == 'mixstyle_kl_mmd':
        params['consistency_lambda'] = 2.0
        params['mmd_lambda'] = 0.1
    return ModelConfig(**params)


def create_manifest(index_df, split, site, split_seed, root):
    from hashlib import sha256
    parts = []
    for label, key in [('train','train_df'), ('validation','val_df'),
                       ('in_domain_test','in_domain_test_df'),
                       ('out_domain_test','out_domain_test_df')]:
        frame = split[key][['site','case_id']].drop_duplicates().copy()
        frame['split'] = label
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True).sort_values(['split', 'site', 'case_id'])
    data = result.to_csv(index=False)
    checksum = sha256(data.encode()).hexdigest()
    output = root / f'split_site{site}_splitseed{split_seed}.csv'
    if output.exists():
        old = output.read_text()
        if sha256(old.encode()).hexdigest() != checksum:
            raise RuntimeError(f'Existing split manifest differs from freshly generated split: {output}')
    else:
        output.write_text(data)
    print(f'Split: {output}, sha256={checksum[:20]}',flush=True)
    # Check site isolation and patient leakage explicitly.
    assert set(result.loc[result.split=='out_domain_test','site']) == {site}
    assert site not in set(result.loc[result.split!='out_domain_test','site'])
    assert not result.case_id.duplicated().any(), 'A case occurs in multiple splits'
    return checksum


def evaluate(cfg, data_cfg, df, loaders, site, seed, split_seed, method, split_hash):
    from dg.experiment import load_trained_model
    from dg.training import eval_metrics
    model = load_trained_model(cfg)
    a = eval_metrics(model, loaders['test_loader'])
    b = eval_metrics(model, loaders['out_domain_test_loader'])
    del model
    return dict(method=method, name=cfg.name, site=site, train_seed=seed,
                split_seed=split_seed, split_sha256=split_hash,
                in_bal_acc=a['balanced_accuracy'], ood_bal_acc=b['balanced_accuracy'],
                in_acc=a['accuracy'], ood_acc=b['accuracy'],
                ood_tumor_f1=b['tumor_f1'], ood_tumor_precision=b['tumor_precision'],
                ood_tumor_recall=b['tumor_recall'],
                gap=a['balanced_accuracy']-b['balanced_accuracy'],
                in_tp=a['tp'], in_fp=a['fp'], in_tn=a['tn'], in_fn=a['fn'],
                ood_tp=b['tp'], ood_fp=b['fp'], ood_tn=b['tn'], ood_fn=b['fn'])


def collect_result_files(results_dir):
    files = sorted(results_dir.glob('DG-*_metrics.json'))
    rows = [json.loads(f.read_text()) for f in files]
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(results_dir / 'all_runs.csv', index=False)
        grouped = frame.groupby(['method','site']).agg(
            n=('ood_bal_acc','count'),
            ood_mean=('ood_bal_acc','mean'), ood_std=('ood_bal_acc','std'),
            in_mean=('in_bal_acc','mean'), in_std=('in_bal_acc','std'),
            f1_mean=('ood_tumor_f1','mean'), f1_std=('ood_tumor_f1','std'),
            gap_mean=('gap','mean'),gap_std=('gap','std'))
        grouped.reset_index().to_csv(results_dir / 'by_site_summary.csv', index=False)
        print(grouped.round(4).to_string(), flush=True)
        # Macro-average over held-out sites, each site weighted equally.
        by_method = grouped.groupby(level=0).agg(
            sites=('n','count'), site_macro_ood=('ood_mean','mean'),
            site_macro_f1=('f1_mean','mean'), site_macro_gap=('gap_mean','mean'))
        by_method.reset_index().to_csv(results_dir / 'by_method_macro_summary.csv', index=False)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent / 'brats23_dg_png')
    p.add_argument('--index', type=Path)
    p.add_argument('--sites', type=int, nargs='+', default=DEFAULT_SITES)
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--split-seed', type=int, default=18662)
    p.add_argument('--methods', choices=METHODS, nargs='+', default=METHODS)
    p.add_argument('--csu-p', type=float, default=0.75)
    p.add_argument('--csu-alpha', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    args=p.parse_args()
    if args.eval_only and args.force:
        p.error('Choose either --eval-only or --force')
    if not (0 <= args.csu_p <= 1 and args.csu_alpha > 0):
        p.error('CSU p must be between 0 and 1; alpha must be >0')
    if len(set(args.sites)) != len(args.sites) or len(set(args.seeds)) != len(args.seeds):
        p.error('Repeated sites or seeds are not allowed')
    if any(s < 0 for s in args.seeds):
        p.error('Seeds must be non-negative')

    from dg import config as dgconfig
    from dg.experiment import run_experiment, prepare_dataloaders_from_config
    index, _=load_index(args.data_root, args.index)
    root=Path(dgconfig.RESULTS_DIR) / 'multisite_runs'
    root.mkdir(parents=True, exist_ok=True)
    logs=Path(dgconfig.RESULTS_DIR).parent / 'logs' / 'multisite_runs'
    logs.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not args.eval_only:
        require_training_device(args.allow_cpu)
    print('Sites:',args.sites,'Training seeds:',args.seeds,'Fixed split seed:',args.split_seed,flush=True)
    print('Methods:',args.methods, 'CSU p/alpha:',args.csu_p,args.csu_alpha,flush=True)

    for site in args.sites:
        data_cfg, split = create_data_split(index, site, args.split_seed)
        split_hash = create_manifest(index,split,site,args.split_seed,root)
        if args.dry_run:
            continue
        loaders=None
        for method in args.methods:
            for seed in args.seeds:
                cfg=config_for(method,site,seed,args.split_seed,args.csu_p,args.csu_alpha)
                metrics_path=root / (cfg.name+'_metrics.json')
                ckpt=Path(dgconfig.CHECKPOINT_DIR) / f'best_{cfg.name}.pth'
                if metrics_path.exists() and not args.force:
                    row=json.loads(metrics_path.read_text())
                    if row['split_sha256'] != split_hash:
                        raise RuntimeError(f'Existing result uses different split: {metrics_path}')
                    print('Skipping completed:', cfg.name,flush=True)
                    continue
                if args.eval_only and not ckpt.exists():
                    print('Skipping absent checkpoint:',ckpt,flush=True)
                    continue
                log=logs / (cfg.name+'.log')
                mode='w' if args.force or not log.exists() else 'a'
                with log.open(mode, buffering=1) as dest:
                    with contextlib.redirect_stdout(Tee(sys.stdout, dest)), contextlib.redirect_stderr(Tee(sys.stderr,dest)):
                        print('Experiment:', cfg.name, flush=True)
                        print('Split SHA256:',split_hash, flush=True)
                        if not args.eval_only and (args.force or not ckpt.exists()):
                            run_experiment(cfg,data_cfg,index,epochs=args.epochs,
                                           patience=args.patience,seed=seed)
                        else:
                            print('Evaluating existing checkpoint:',ckpt,flush=True)
                        if loaders is None:
                            loaders=prepare_dataloaders_from_config(data_cfg,index_df=index,verbose=False)
                        row=evaluate(cfg,data_cfg,index,loaders,site,seed,args.split_seed,method,split_hash)
                        metrics_path.write_text(json.dumps(row,indent=2))
                        print(pd.DataFrame([row])[['method','site','train_seed','in_bal_acc','ood_bal_acc','ood_tumor_f1','gap']].round(4).to_string(index=False),flush=True)
                collect_result_files(root)
    if not args.dry_run:
        print('Results:',root,flush=True)
        collect_result_files(root)

if __name__=='__main__':
    main()
