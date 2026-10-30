#!/usr/bin/env python3
"""Validate/select MixStyle + Multi by SOURCE validation; show per-site OOD.

Usage (from AWS brats_dg project root):
  python plot_multiscale_validation_selected.py
  python plot_multiscale_validation_selected.py --regimes strong weak

The script never uses an OOD metric to choose a (regime, w) pair.
The default graph displays a validation-selected weight within the STRONG
MixStyle regime, making a more direct comparison with strong MixStyle baselines.
An optional second Multi series shows validation-selected WEAK regime weights.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPECTED_SEEDS = {0,1,2,3,4}
METHOD_LABELS = {
    'erm': 'ResNet-18',
    'mixstyle': 'MixStyle (strong)',
    'csu': 'CSU',
    'mixstyle_kl': 'MixStyle + KL',
    'mixstyle_kl_mmd': 'MixStyle + KL + MMD',
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing', type=Path,
                        default=Path('artifacts/results/multisite_runs/all_runs.csv'))
    parser.add_argument('--multi', type=Path,
                        default=Path('artifacts/results/multiscale_runs/multi_all_runs.csv'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('artifacts/results/multiscale_runs'))
    parser.add_argument('--regimes', nargs='+', choices=('strong','weak'),
                        default=['strong'], help='Select best w within EACH named regime, independently.')
    parser.add_argument('--sites', nargs='+', type=int, default=[1,4,13,18,20])
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Exploratory use only: permit configurations missing some training seeds.')
    args = parser.parse_args()
    if not args.existing.is_file() or not args.multi.is_file():
        parser.error('Missing input CSV; run the sweep first. Expected:\n '
                     f'{args.existing}\n {args.multi}')
    old = pd.read_csv(args.existing)
    multi = pd.read_csv(args.multi)
    need_old = {'site','method','train_seed','ood_bal_acc','in_bal_acc','split_sha256'}
    need_new = {'site','regime','multi_weight','train_seed','best_val','ood_bal_acc',
                'in_bal_acc','split_sha256'}
    if need_old - set(old) or need_new - set(multi):
        raise ValueError(f'Missing columns: old={need_old-set(old)} multi={need_new-set(multi)}')
    old = old.loc[old.site.isin(args.sites)].copy()
    multi = multi.loc[multi.site.isin(args.sites) & multi.regime.isin(args.regimes)].copy()
    if multi.empty:
        raise ValueError('No Multi rows for the requested sites/regimes')
    if multi.best_val.isna().any():
        raise ValueError('Multi best_val values missing. Need source validation scores to select w.')
    if multi.duplicated(['site','regime','multi_weight','train_seed']).any():
        raise ValueError('Duplicate Multi runs for a site/regime/weight/seed')
    if old.duplicated(['site','method','train_seed']).any():
        raise ValueError('Duplicate rows in existing 125-run CSV')
    # Confirm Multi uses the same case-wise data split as old baselines.
    for site in args.sites:
        prior_sha = set(old.loc[old.site.eq(site),'split_sha256'])
        mult_sha = set(multi.loc[multi.site.eq(site),'split_sha256'])
        if prior_sha and mult_sha and (len(prior_sha)!=1 or len(mult_sha)!=1 or prior_sha!=mult_sha):
            raise ValueError(f'Split mismatch for site {site}: old={prior_sha} vs Multi={mult_sha}')

    results = []
    incomplete = []
    for (site,regime,weight), g in multi.groupby(['site','regime','multi_weight']):
        seeds = set(map(int, g.train_seed))
        if not args.allow_incomplete and seeds!=EXPECTED_SEEDS:
            incomplete.append((site,regime,weight,sorted(seeds)))
            continue
        results.append({'site':site,'regime':regime,'multi_weight':weight,
                        'n':len(g),'mean_source_val_acc':g.best_val.mean(),
                        'source_val_std':g.best_val.std(ddof=1)})
    if incomplete:
        print(f'Ignoring {len(incomplete)} incomplete site/regime/w combinations; examples:',
              incomplete[:4],file=sys.stderr)
    if not results:
        raise RuntimeError('No completed sweep groups. Finish all five seeds per configuration.')
    cfg = pd.DataFrame(results)
    # Select each weight entirely via source-domain validation accuracy.
    # Ties: prefer smaller w for a reproducible deterministic rule.
    cfg=cfg.sort_values(['site','regime','mean_source_val_acc','multi_weight'],
                        ascending=[True,True,False,True])
    selected=cfg.drop_duplicates(['site','regime']).copy()
    selected['selection_rule']='max mean SOURCE val accuracy; tie -> lower weight'
    summary=[]
    for row in selected.itertuples(index=False):
        g = multi[(multi.site==row.site) & (multi.regime==row.regime) &
                  np.isclose(multi.multi_weight,row.multi_weight)]
        summary.append({'site':row.site, 'method':f'MixStyle + Multi ({row.regime})',
                        'selected_w':row.multi_weight,'n':len(g),
                        'source_val_acc':row.mean_source_val_acc,
                        'ood_mean':g.ood_bal_acc.mean(),
                        'ood_std':g.ood_bal_acc.std(ddof=1),
                        'in_mean':g.in_bal_acc.mean(),
                        'in_std':g.in_bal_acc.std(ddof=1)})
    multi_summary=pd.DataFrame(summary)
    old_summ=old.groupby(['site','method'],as_index=False).agg(
        n=('ood_bal_acc','size'),ood_mean=('ood_bal_acc','mean'),
        ood_std=('ood_bal_acc','std'),in_mean=('in_bal_acc','mean'),
        in_std=('in_bal_acc','std'))
    old_summ['method']=old_summ['method'].map(METHOD_LABELS).fillna(old_summ['method'])
    old_summ['selected_w']=np.nan
    old_summ['source_val_acc']=np.nan
    combined=pd.concat([old_summ,multi_summary],ignore_index=True,sort=False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg.to_csv(args.output_dir/'source_validation_weight_sweep.csv',index=False)
    selected.to_csv(args.output_dir/'source_validation_selected_weights.csv',index=False)
    combined.to_csv(args.output_dir/'per_site_comparison_with_selected_multi.csv',index=False)
    sites=sorted(set(args.sites))
    fig,ax=plt.subplots(figsize=(11,6.1))
    for method,g in combined.groupby('method',sort=False):
        g=g.set_index('site').reindex(sites)
        ax.errorbar(sites,100*g['ood_mean'],yerr=100*g['ood_std'],label=method,
                    marker='o',capsize=2,linewidth=1.4)
    ax.set_xticks(sites,[f'Site {s}' for s in sites])
    ax.set_xlabel('Held-out clinical site')
    ax.set_ylabel('OOD balanced accuracy (%)')
    ax.set_title('Per-site OOD accuracy: source-validation-selected Multi weights\nMean ± SD across five training seeds')
    ax.grid(axis='y',alpha=0.25)
    ax.legend(loc='best',fontsize=8,ncol=2)
    fig.tight_layout()
    png=args.output_dir/'ood_comparison_with_val_selected_multi.png'
    fig.savefig(png,dpi=180)
    pdf=args.output_dir/'ood_comparison_with_val_selected_multi.pdf'
    fig.savefig(pdf)
    plt.close(fig)
    print('\nSelected weights (selected using SOURCE validation accuracy ONLY):')
    print(selected[['site','regime','multi_weight','n','mean_source_val_acc']].to_string(index=False))
    print('\nMulti OOD accuracy after selection (NOT used in selection):')
    print(multi_summary.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nSaved:')
    for path in [args.output_dir/'source_validation_weight_sweep.csv',
                 args.output_dir/'source_validation_selected_weights.csv',
                 args.output_dir/'per_site_comparison_with_selected_multi.csv',png,pdf]:
        print(' ',path)

if __name__=='__main__':
    main()
