#!/usr/bin/env python3
"""Staged single-source-site-held-out validation for BraTS multi-site DG.

Runs against the existing AWS `dg/` package; does not overwrite previous 125 runs.
Stages: style -> kl -> joint -> multi -> final. Fixed ResNet-18 backbone (no LR sweep).
Balanced accuracy on a fixed sample from ONE ENTIRELY HELD-OUT SOURCE SITE is the
selection metric. Target image data is opened ONLY during final evaluation.
One site is excluded from training; up to 64 fixed cases represent it for validation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from aws_paths import load_index, create_data_split, require_training_device

SITES = (18, 1, 4, 13, 20)
# Predeclared based on site sizes (not on evaluation scores of this study).
VALIDATION_SITE_BY_TARGET = {18: 1, 1: 18, 4: 1, 13: 1, 20: 1}
DEFAULT_VAL_CASE_CAP = 64
SEARCH_SEEDS = (0, 1)
FINAL_SEEDS = (100, 101, 102, 103, 104)
STAGES = ('style', 'kl', 'joint', 'multi', 'final')
METHODS = ('erm', 'mixstyle', 'csu', 'mixstyle_kl', 'mixstyle_kl_mmd', 'mixstyle_multi')
STYLE_P = (0.25, 0.5, 0.75)
STYLE_A = (0.1, 0.3, 0.5)
# Fixed Adam learning rate from the existing dg/experiment.py and prior BraTS studies.
# NOT optimized separately for any model or held-out target site.
FIXED_BACKBONE_LR = 1e-4
KL_VALUES = (0.5, 1.0, 2.0, 3.0)
JOINT_KL = (0.5, 1.0, 2.0)
JOINT_MMD = (0.025, 0.05, 0.1, 0.25)
MULTI_REGIMES = {'strong': (0.75, 0.1), 'weak': (0.5, 0.5)}
MULTI_W = (0.25, 0.5, 1.0, 2.0, 4.0)

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, message):
        for stream in self.streams: stream.write(message)
        return len(message)
    def flush(self):
        for stream in self.streams: stream.flush()

def fmt(v):
    return f'{v:g}' if isinstance(v, (float, int)) and not isinstance(v, bool) else str(v)

def params_key(params):
    return '__'.join(f'{key}{fmt(value)}' for key, value in sorted(params.items()))

def trial_name(stage, method, site, val_site, val_cap, seed, split_seed, params):
    return (f'SITEVAL-{stage}-{method}-target{site}-val{val_site}-cap{val_cap}'
            f'-split{split_seed}-seed{seed}__{params_key(params)}')

def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)

def split_checksum(split):
    parts = []
    for label, key in [('train','train_df'),('validation','val_df'),('in_domain_test','in_domain_test_df'),('out_domain_test','out_domain_test_df')]:
        x = split[key][['site','case_id']].drop_duplicates().copy()
        x['split'] = label
        parts.append(x)
    s = pd.concat(parts, ignore_index=True).sort_values(['split','site','case_id']).to_csv(index=False)
    return hashlib.sha256(s.encode()).hexdigest()



def validation_site_for_target(target: int, override: dict[int,int]) -> int:
    if target not in override:
        raise ValueError(f'No predeclared validation site for target {target}. '
                         'Provide a mapping with --validation-map, e.g. 18:1 1:18 4:1')
    site = override[target]
    if site == target:
        raise ValueError(f'Validation site {site} must differ from target site {target}')
    return site


def create_source_site_holdout_split(index_df, target_site: int, val_site: int,
                                     seed: int, val_cap: int = DEFAULT_VAL_CASE_CAP):
    """Outer target test; one entirely excluded source-site validation domain.

    Start from the existing source patient split for comparability and stability:
    - original train+source-val patients on OTHER source sites become training;
    - existing in-domain test patients on OTHER source sites remain test;
    - NO patients from the validation site enter training or in-domain test;
    - evaluate up to val_cap cases chosen deterministically from the validation site;
    - target cases remain an untouched outer test site.

    Validation-site patients not selected into the val_cap are entirely unused.
    There is only ONE model trained per hyperparameter/seed, not extra inner folds.
    """
    from dg.experiment import DataSplitConfig
    if target_site == val_site or val_cap < 1:
        raise ValueError('Require a distinct target and validation site, and val_cap >= 1')
    sites = set(index_df['site'].astype(int).unique())
    if not {int(target_site),int(val_site)}.issubset(sites):
        raise ValueError(f'Target/validation site missing from index; available: {sorted(sites)}')
    _, base = create_data_split(index_df, target_site, seed)
    case_site = index_df[['case_id','site']].drop_duplicates().copy()
    if case_site['case_id'].duplicated().any():
        raise RuntimeError('case_id appears in multiple site records; requires compound site/case identifier')
    val_candidates = (case_site.loc[case_site['site'].eq(val_site), 'case_id']
                      .sort_values(key=lambda col: col.astype(str)).reset_index(drop=True))
    # Fixed shuffle/subsample independent of model, training seed, and held-out target labels.
    val_ids = set(val_candidates.sample(n=min(len(val_candidates),val_cap),random_state=seed+int(val_site)).tolist())
    train_ids = (set(base['train_set']) | set(base['val_set']))
    in_ids = set(base['in_domain_test_set'])
    banned = set(val_candidates.tolist())
    train_ids.difference_update(banned)
    in_ids.difference_update(banned)
    target_ids = set(base['out_domain_test_set'])
    data_cfg = DataSplitConfig(name=f'target{target_site}_sourceval{val_site}_cap{val_cap}',
                               train_cases=frozenset(train_ids),
                               validation_cases=frozenset(val_ids),
                               test_cases=frozenset(in_ids),
                               out_domain_test_cases=frozenset(target_ids))
    split = {
        'train_df': index_df[index_df['case_id'].isin(train_ids)].copy(),
        'val_df': index_df[index_df['case_id'].isin(val_ids)].copy(),
        'in_domain_test_df': index_df[index_df['case_id'].isin(in_ids)].copy(),
        'out_domain_test_df': index_df[index_df['case_id'].isin(target_ids)].copy(),
    }
    return data_cfg, split

def validate_site_split(split, site, val_site):
    names = ('train_df','val_df','in_domain_test_df','out_domain_test_df')
    case_sets = [set(split[key]['case_id'].astype(str)) for key in names]
    for i in range(len(case_sets)):
        for j in range(i+1, len(case_sets)):
            if case_sets[i].intersection(case_sets[j]):
                raise RuntimeError(f'Patient leakage: {names[i]} versus {names[j]}')
    if set(split['out_domain_test_df']['site'].unique()) != {site}:
        raise RuntimeError(f'OOD site mismatch for {site}')
    for key in names[:-1]:
        if site in set(split[key]['site'].unique()):
            raise RuntimeError(f'Target site {site} leaked into {key}')
    val_sites = set(split['val_df']['site'].unique())
    if val_sites != {val_site}:
        raise RuntimeError(f'Validation must exclusively use site {val_site}, observed {val_sites}')
    for key in ('train_df','in_domain_test_df','out_domain_test_df'):
        if val_site in set(split[key]['site'].unique()):
            raise RuntimeError(f'Source validation site {val_site} leaked into {key}')
    for key in names:
        if split[key].empty:
            raise RuntimeError(f'Empty {key} for target {site}, source validation {val_site}')
    return split_checksum(split)

def all_specs_for_stage(stage, choices):
    """Yield (method, hyperparameters), with dependencies source-selected."""
    lr = FIXED_BACKBONE_LR
    if stage == 'style':
        for method in ('mixstyle', 'csu'):
            for p in STYLE_P:
                for alpha in STYLE_A:
                    yield method, {'lr':lr, 'p':p, 'alpha':alpha}
    elif stage == 'kl':
        base = choices['mixstyle']
        for lam in KL_VALUES:
            yield 'mixstyle_kl', {'lr':lr, 'p':base['p'], 'alpha':base['alpha'], 'lambda_c':lam}
    elif stage == 'joint':
        base = choices['mixstyle']
        for kl in JOINT_KL:
            for mmd in JOINT_MMD:
                yield 'mixstyle_kl_mmd', {'lr':lr, 'p':base['p'], 'alpha':base['alpha'], 'lambda_c':kl, 'lambda_m':mmd}
    elif stage == 'multi':
        # Matches the previously declared strong/weak MixStyle regimes.
        for regime, (p, alpha) in MULTI_REGIMES.items():
            for weight in MULTI_W:
                yield 'mixstyle_multi', {'lr':lr, 'p':p, 'alpha':alpha, 'w':weight, 'regime':regime}
    else:
        raise ValueError(stage)

def model_config(method, params, name):
    from dg.experiment import ModelConfig
    kwargs = dict(name=name, train_layers=(2,))
    if method in ('mixstyle', 'mixstyle_kl', 'mixstyle_kl_mmd', 'mixstyle_multi'):
        kwargs.update(use_mixstyle=True, mixstyle_p=float(params['p']), mixstyle_a=float(params['alpha']))
    elif method == 'csu':
        kwargs.update(use_csu=True, csu_p=float(params['p']), csu_alpha=float(params['alpha']))
    if method in ('mixstyle_kl', 'mixstyle_kl_mmd'):
        kwargs['consistency_lambda'] = float(params['lambda_c'])
    if method == 'mixstyle_kl_mmd':
        kwargs['mmd_lambda'] = float(params['lambda_m'])
    return ModelConfig(**kwargs)

def train_one(cfg, method, params, data_cfg, index, seed, path, epochs, patience):
    import torch
    from dg import runtime
    from dg.experiment import build_model, prepare_dataloaders_from_config
    from dg.training import (set_seed, eval_metrics, train_epoch, train_epoch_consistency, train_epoch_kl_mmd)
    set_seed(seed)
    runtime.reset_scaler()
    # Original loader eagerly caches all NPZs, including OOD. Suppress target NPZ
    # loading for ALL training/validation, including the final-seed stage.
    source_only_cfg = replace(data_cfg, out_domain_test_cases=frozenset())
    loaders = prepare_dataloaders_from_config(source_only_cfg, index_df=index, verbose=False)
    model, optimizer, criterion = build_model(cfg, pretrained=True, verbose=False)
    for group in optimizer.param_groups:
        group['lr'] = float(params['lr'])
    best_val = -float('inf')
    best_epoch = None
    no_improve = 0
    history = []
    for epoch in range(1, epochs+1):
        if method == 'mixstyle_kl':
            t = train_epoch_consistency(model, loaders['train_loader'], optimizer, criterion, lambda_c=params['lambda_c'])
            tfields = dict(train_loss=t[0], ce=t[1], kl=t[2], train_acc=t[3])
        elif method == 'mixstyle_kl_mmd':
            t = train_epoch_kl_mmd(model, loaders['train_loader'], optimizer, criterion,
                                   lambda_c=params['lambda_c'], lambda_mmd=params['lambda_m'])
            tfields = dict(train_loss=t[0], ce=t[1], kl=t[2], mmd=t[3], train_acc=t[4])
        elif method == 'mixstyle_multi':
            try:
                from dg.multiscale import train_epoch_multiscale
            except ImportError as e:
                raise RuntimeError('Missing dg/multiscale.py; install MixStyle_Multi_AWS_patch.zip first') from e
            t = train_epoch_multiscale(model, loaders['train_loader'], optimizer, criterion, weight=params['w'])
            tfields = {f'train_{k}': v for k, v in t.items()}
        else:
            t = train_epoch(model, loaders['train_loader'], optimizer, criterion)
            tfields = dict(train_loss=t[0], train_acc=t[1])
        # This uses ONLY the source validation cases. No OOD loader accessed in tuning.
        valid = eval_metrics(model, loaders['val_loader'])
        val_bal = float(valid['balanced_accuracy'])
        hist = dict(epoch=epoch, val_bal_acc=val_bal, val_acc=float(valid['accuracy']), **tfields)
        history.append(hist)
        print(f'{cfg.name}: epoch {epoch}/{epochs} | source-val balanced={val_bal:.5f}, acc={valid["accuracy"]:.5f}', flush=True)
        if val_bal > best_val + 1e-12:
            best_val = val_bal
            best_epoch = epoch
            torch.save(model.state_dict(), path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Patience {patience} exhausted at epoch {epoch}',flush=True)
                break
    del model, optimizer, criterion, loaders
    return best_val, best_epoch, history

def eval_checkpoint(cfg, ckpt, data_cfg, index):
    import torch
    from dg import runtime
    from dg.experiment import build_model, prepare_dataloaders_from_config
    from dg.training import eval_metrics
    loaders = prepare_dataloaders_from_config(data_cfg, index_df=index, verbose=False)
    model, _, _ = build_model(cfg, pretrained=False, verbose=False)
    model.load_state_dict(torch.load(ckpt, map_location=runtime.DEVICE, weights_only=True))
    model.eval()
    a = eval_metrics(model, loaders['test_loader'])
    b = eval_metrics(model, loaders['out_domain_test_loader'])
    del model, loaders
    return a, b

def paths(root, stage, method, name):
    # Root is the project artifacts folder.
    base = root/'results'/f'siteholdout_{stage}'/method
    return dict(metrics=base/f'{name}_metrics.json',
                history=base/f'{name}_history.json',
                ckpt=root/'checkpoints'/f'siteholdout_{stage}'/method/f'best_{name}.pth',
                log=root/'logs'/f'siteholdout_{stage}'/method/f'{name}.log')

def run_trial(root, stage, method, site, val_site, val_cap, seed, split_seed, params, data_cfg, index, checksum, epochs, patience, dry_run=False):
    name = trial_name(stage, method, site, val_site, val_cap, seed, split_seed, params)
    dest = paths(root, stage, method, name)
    if dry_run:
        print('PLAN',name, flush=True)
        return
    if dest['metrics'].exists():
        old = json.loads(dest['metrics'].read_text())
        if old.get('split_sha256') != checksum:
            raise RuntimeError(f'Existing run has different split: {dest["metrics"]}')
        if old.get('parameters') != params:
            raise RuntimeError(f'Existing run uses different hyperparameters: {name}')
        print('SKIP completed',name,flush=True)
        return
    for path in dest.values(): path.parent.mkdir(parents=True, exist_ok=True)
    with dest['log'].open('a', buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            print('BEGIN',name, flush=True)
            print('RUN',json.dumps({'stage':stage,'site':site,'val_site':val_site,'val_case_cap':val_cap,'seed':seed,'params':params,'split_sha256':checksum}),flush=True)
            cfg = model_config(method, params, name)
            # Incomplete checkpoints must not silently count as complete searches.
            # Explicit reruns from scratch if no _metrics.json is present.
            best_val, best_epoch, history = train_one(cfg,method,params,data_cfg,index,seed,
                                                       dest['ckpt'],epochs,patience)
            save_json_atomic(dest['history'], dict(history=history, best_val_bal_acc=best_val,
                                                   best_epoch=best_epoch, split_sha256=checksum))
            row = dict(name=name, method=method, stage=stage, site=site, val_site=val_site, val_case_cap=val_cap, train_seed=seed,
                       split_seed=split_seed, split_sha256=checksum,
                       parameters=params, best_val_bal_acc=best_val, best_epoch=best_epoch,
                       checkpoint=str(dest['ckpt']))
            if stage == 'final':
                ind, ood = eval_checkpoint(cfg, dest['ckpt'], data_cfg,index)
                row.update(in_bal_acc=ind['balanced_accuracy'], in_acc=ind['accuracy'],
                           ood_bal_acc=ood['balanced_accuracy'],ood_acc=ood['accuracy'],
                           ood_tumor_f1=ood['tumor_f1'],
                           ood_tumor_precision=ood['tumor_precision'],
                           ood_tumor_recall=ood['tumor_recall'],
                           gap=ind['balanced_accuracy']-ood['balanced_accuracy'])
            save_json_atomic(dest['metrics'],row)
            if stage == 'final':
                print(f'FINISHED {name}: val={best_val:.5f}, in_bal={row["in_bal_acc"]:.5f}, ood_bal={row["ood_bal_acc"]:.5f}',flush=True)
            else:
                print(f'FINISHED {name}: val={best_val:.5f} (no target evaluation)',flush=True)

def all_metrics(root, stage):
    files = sorted((root/'results'/f'siteholdout_{stage}').glob('*/*_metrics.json'))
    return [json.loads(f.read_text()) for f in files]

def selected_path(root):
    return root/'results'/'siteholdout_search'/'selected_hparams.json'

def load_selected(root):
    path = selected_path(root)
    if not path.is_file(): return {}
    data = json.loads(path.read_text())
    # A legacy ResNet learning-rate search is never reused in this fixed-LR protocol.
    for per_site in data.values():
        if isinstance(per_site, dict): per_site.pop('erm', None)
    return data

def select_stage(root, stage, sites, seeds, expected, split_seed, hashes):
    """Require all configured search seeds for every candidate before selecting."""
    selected = load_selected(root)
    rows = all_metrics(root, stage)
    method_groups = sorted({name for name, _ in expected})
    for site in sites:
        protocol_file = root/'results'/'siteholdout_search'/f'protocol_site{site}.json'
        checksum, val_site, val_cap = hashes[site]
        protocol = dict(split_sha256=checksum,split_seed=split_seed,
                        val_site=val_site,val_case_cap=val_cap,
                        search_seeds=sorted(seeds),selection_metric='held_out_source_site_balanced_accuracy',
                        fixed_backbone_lr=FIXED_BACKBONE_LR,backbone_lr_tuned=False)
        if protocol_file.is_file() and json.loads(protocol_file.read_text()) != protocol:
            raise RuntimeError(f'Existing tuning protocol differs for site {site}: {protocol_file}')
        choices = selected.setdefault(str(site),{})
        for method in method_groups:
            plans = [p for m,p in expected if m == method]
            scores = []
            for params in plans:
                matches = [r for r in rows if r.get('site')==site and r.get('val_site')==val_site
                           and r.get('val_case_cap')==val_cap and r.get('split_sha256')==checksum
                           and r.get('method')==method and r.get('parameters')==params and r.get('train_seed') in seeds]
                actual_seeds = sorted({r['train_seed'] for r in matches})
                if actual_seeds != sorted(seeds) or len(matches) != len(seeds):
                    print(f'INCOMPLETE search, no selection yet: site={site}, method={method}, params={params} ({len(matches)}/{len(seeds)})',flush=True)
                    scores = []
                    break
                mean = sum(float(r['best_val_bal_acc']) for r in matches) / len(matches)
                scores.append((mean,params))
            if scores:
                scores.sort(key=lambda v:(-v[0],params_key(v[1])))
                best_score, best_params = scores[0]
                choices[method] = dict(best_params)
                save_json_atomic(protocol_file, protocol)
                print(f'SELECT source-only site={site} {method}: {best_params}, mean val_bal={best_score:.5f}',flush=True)
        # Existing selections on sites not part of this command stay unchanged.
    save_json_atomic(selected_path(root),selected)
    # Human-readable table, including separate site/method/config validation ranking.
    flat = [dict(site=int(site),method=method, **params) for site, choices in selected.items() for method,params in choices.items()]
    if flat:
        pd.DataFrame(flat).sort_values(['site','method']).to_csv(root/'results'/'siteholdout_search'/'selected_hparams.csv',index=False)
    return selected

def final_report(root):
    rows = all_metrics(root,'final')
    if not rows: return
    folder = root/'results'/'siteholdout_final'
    flat = [{k:v for k,v in r.items() if k!='parameters'} | r['parameters'] for r in rows]
    df = pd.DataFrame(flat).sort_values(['method','site','train_seed'])
    df.to_csv(folder/'all_final_runs.csv', index=False)
    gb = df.groupby(['method','site']).agg(n=('ood_bal_acc','size'),
            in_mean=('in_bal_acc','mean'),in_std=('in_bal_acc','std'),
            ood_mean=('ood_bal_acc','mean'),ood_std=('ood_bal_acc','std'),
            f1_mean=('ood_tumor_f1','mean'),f1_std=('ood_tumor_f1','std'),
            gap_mean=('gap','mean'),gap_std=('gap','std')).reset_index()
    gb.to_csv(folder/'per_site_comparison.csv',index=False)
    print('\nFINAL PER SITE (not macro-average):\n'+gb.round(4).to_string(index=False),flush=True)
    # Main requested visual: one OOD line for each method, without disclosing weight in legend.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,5.5))
        labels={'erm':'ResNet-18','mixstyle':'MixStyle','csu':'CSU',
                'mixstyle_kl':'MixStyle + KL',
                'mixstyle_kl_mmd':'MixStyle + KL + MMD',
                'mixstyle_multi':'MixStyle + Multi'}
        for method, part in gb.groupby('method',sort=False):
            part=part.sort_values('site')
            ax.errorbar(part['site'].astype(str),part['ood_mean']*100,
                        yerr=part['ood_std'].fillna(0)*100,marker='o',capsize=3,label=labels.get(method,method))
        ax.set_ylabel('OOD balanced accuracy (%)')
        ax.set_xlabel('Held-out clinical site')
        ax.set_title('Source-validation-tuned models (independent evaluation seeds)')
        ax.legend(loc='best',fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(folder/f'per_site_ood_tuned.{ext}', dpi=180)
        plt.close(fig)
    except ImportError:
        print('matplotlib not installed: plot skipped; CSVs saved.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True,choices=STAGES)
    p.add_argument('--sites',type=int,nargs='+',default=list(SITES))
    p.add_argument('--search-seeds',type=int,nargs='+',default=list(SEARCH_SEEDS))
    p.add_argument('--final-seeds',type=int,nargs='+',default=list(FINAL_SEEDS))
    p.add_argument('--split-seed',type=int,default=18662)
    p.add_argument('--val-case-cap',type=int,default=DEFAULT_VAL_CASE_CAP,
                   help='Use at most this many cases from the source validation site; exclude all its remaining cases from training.')
    p.add_argument('--validation-map',nargs='+',default=['18:1','1:18','4:1','13:1','20:1'],
                   help='One designated validation site per OOD site: 18:1 1:18 4:1 13:1 20:1')
    p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent/'brats23_dg_png')
    p.add_argument('--index',type=Path)
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS),help='Final stage only')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    args=p.parse_args()
    if not args.sites or not args.search_seeds or not args.final_seeds:
        p.error('sites, search seeds, and final seeds must be nonempty')
    if len(set(args.sites))!=len(args.sites) or len(set(args.search_seeds))!=len(args.search_seeds) or len(set(args.final_seeds))!=len(args.final_seeds):
        p.error('duplicate sites or seeds')
    if set(args.search_seeds) & set(args.final_seeds):
        p.error('Search and final evaluation seeds must be disjoint')
    if args.epochs<1 or args.patience<1 or args.val_case_cap<1:
        p.error('epochs, patience and val-case-cap must be >= 1')
    try:
        val_map = dict((int(item.split(':')[0]), int(item.split(':')[1])) for item in args.validation_map)
    except (ValueError, IndexError) as e:
        p.error(f'--validation-map must contain target:source pairs, got {args.validation_map}: {e}')
    if len(val_map) != len(args.validation_map):
        p.error('Duplicate target sites in --validation-map')
    if not args.dry_run:
        require_training_device(args.allow_cpu)
    from dg import config as cfg
    root=Path(cfg.ARTIFACTS_ROOT)
    # Create CSV/report parents before writing split manifests or selections.
    # Earlier versions only created per-run results directories.
    for folder in ('siteholdout_search', 'siteholdout_final'):
        (root / 'results' / folder).mkdir(parents=True, exist_ok=True)
    data,_=load_index(args.data_root,args.index)
    selected=load_selected(root)
    print('FIXED backbone: ResNet-18 L2+FC, Adam learning rate', FIXED_BACKBONE_LR, '(NO learning-rate search)', flush=True)
    print('Stage:',args.stage,'sites:',args.sites,'search seeds:',args.search_seeds,
          'final seeds:',args.final_seeds,'split seed:',args.split_seed,flush=True)
    for site in args.sites:
        val_site=validation_site_for_target(site,val_map)
        data_cfg, split=create_source_site_holdout_split(data,site,val_site,args.split_seed,args.val_case_cap)
        checksum=validate_site_split(split,site,val_site)
        print(f'TARGET={site} VALIDATION_SITE={val_site} (cases={len(data_cfg.validation_cases)}, cap={args.val_case_cap})',flush=True)
        print('Split sizes: train=',len(data_cfg.train_cases),'source-val=',len(data_cfg.validation_cases),
              'in-domain test=',len(data_cfg.test_cases),'OOD=',len(data_cfg.out_domain_test_cases),flush=True)
        protocol_file = root/'results'/'siteholdout_search'/f'protocol_site{site}.json'
        if protocol_file.is_file():
            prev = json.loads(protocol_file.read_text())
            expected_protocol = dict(split_sha256=checksum, split_seed=args.split_seed,
                                     val_site=val_site, val_case_cap=args.val_case_cap,
                                     search_seeds=sorted(args.search_seeds),selection_metric='held_out_source_site_balanced_accuracy',
                                     fixed_backbone_lr=FIXED_BACKBONE_LR,backbone_lr_tuned=False)
            if prev != expected_protocol:
                raise RuntimeError(f'Existing tuning selection from another split/seed protocol: {protocol_file}')
        manifest= root/'results'/'siteholdout_search'/f'split_site{site}_splitseed{args.split_seed}.json'
        if manifest.is_file() and json.loads(manifest.read_text())['split_sha256']!=checksum:
            raise RuntimeError(f'Split changed from earlier runs: {manifest}')
        if not args.dry_run:
            split_rows=[]
            for label,key in [('train','train_df'),('source_validation','val_df'),
                              ('in_domain_test','in_domain_test_df'),('ood_final_test','out_domain_test_df')]:
                frame=split[key][['site','case_id']].drop_duplicates().copy()
                frame['split']=label
                split_rows.append(frame)
            pd.concat(split_rows,ignore_index=True).sort_values(['split','site','case_id']).to_csv(
                root/'results'/'siteholdout_search'/f'split_site{site}_val{val_site}_cap{args.val_case_cap}.csv',index=False)
            save_json_atomic(manifest,dict(site=site,val_site=val_site,val_case_cap=args.val_case_cap,
                                           split_seed=args.split_seed,split_sha256=checksum,
                                           train_cases=len(data_cfg.train_cases),val_cases=len(data_cfg.validation_cases),
                                           in_test_cases=len(data_cfg.test_cases),ood_cases=len(data_cfg.out_domain_test_cases)))
        if args.stage=='final':
            choices=selected.get(str(site),{})
            for method in args.methods:
                if method == 'erm':
                    # Plain ResNet-18 is a fixed control, not a tuned method.
                    params={'lr': FIXED_BACKBONE_LR}
                else:
                    if method not in choices:
                        raise RuntimeError(f'No source-selected configuration for site {site}, method {method}. Complete its search stage first.')
                    params=choices[method]
                for seed in args.final_seeds:
                    run_trial(root,'final',method,site,val_site,args.val_case_cap,seed,args.split_seed,params,data_cfg,data,checksum,args.epochs,args.patience,args.dry_run)
        else:
            site_choices=selected.get(str(site),{})
            if args.stage in ('kl','joint') and not site_choices.get('mixstyle'):
                raise RuntimeError(f'Stage style must finish first, missing MixStyle selection for site {site}')
            specs=list(all_specs_for_stage(args.stage,site_choices))
            for method,params in specs:
                for seed in args.search_seeds:
                    run_trial(root,args.stage,method,site,val_site,args.val_case_cap,seed,args.split_seed,params,data_cfg,data,checksum,args.epochs,args.patience,args.dry_run)
        if args.stage != 'final' and not args.dry_run:
            selected=select_stage(root,args.stage,[site],args.search_seeds,list(all_specs_for_stage(args.stage,selected.get(str(site),{}))),args.split_seed,{site:(checksum,val_site,args.val_case_cap)})
    if args.stage=='final' and not args.dry_run:
        final_report(root)
    print('Stage complete:',args.stage,flush=True)

if __name__=='__main__':
    main()
