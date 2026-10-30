#!/usr/bin/env python3
"""Two publication-oriented plots for BraTS cross-site DG results.

1. SOURCE-validation accuracy vs. multi-scale loss weight w (per site).
2. OOD balanced accuracy vs. test site for ERM, MixStyle, CSU,
   MixStyle + KL, MixStyle + KL + MMD, and ONE line 'MixStyle + Multi'.

For every held-out site, choose Multi hyperparameters using only SOURCE
validation accuracy averaged across the expected training seeds.
Never select based on OOD scores.

Usage (run from the AWS project root):
    python plot_multiscale_two_views.py
    python plot_multiscale_two_views.py --regimes strong weak

--regimes strong weak allows validation-based selection of both regime AND w;
there is still exactly ONE 'MixStyle + Multi' line in the comparison plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    'erm': 'ResNet-18',
    'mixstyle': 'MixStyle',
    'csu': 'CSU',
    'mixstyle_kl': 'MixStyle + KL',
    'mixstyle_kl_mmd': 'MixStyle + KL + MMD',
}
METHOD_ORDER = list(METHOD_LABELS.values()) + ['MixStyle + Multi']


def verify_columns(df, required, name):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{name} missing CSV columns: {sorted(missing)}')


def validate_seeds(group, expected, description):
    seeds = list(group['train_seed'].astype(int))
    return set(seeds) == expected and len(seeds) == len(expected)


def create_figures(args):
    expected_seeds = set(args.seeds)
    existing = pd.read_csv(args.existing)
    multi = pd.read_csv(args.multi)

    required_common = ['site', 'train_seed', 'in_bal_acc', 'ood_bal_acc', 'split_sha256']
    verify_columns(existing, required_common + ['method'], 'Existing')
    verify_columns(multi, required_common + ['regime', 'multi_weight', 'best_val'], 'Multi')
    existing = existing[existing.site.isin(args.sites)].copy()
    multi = multi[multi.site.isin(args.sites) & multi.regime.isin(args.regimes)].copy()
    if len(multi) == 0:
        raise ValueError('No Multi records match --sites and --regimes. Run the sweep first.')
    if multi['best_val'].isna().any():
        raise ValueError('best_val is missing for at least one Multi run. Use source validation scores.')
    if multi.duplicated(['site', 'regime', 'multi_weight', 'train_seed']).any():
        raise ValueError('Duplicate Multi (site, regime, w, train_seed) entries.')
    if existing.duplicated(['site', 'method', 'train_seed']).any():
        raise ValueError('Duplicate baseline (site, method, train_seed) entries.')

    # Confirm that all models were evaluated on exactly the same patient split.
    for site in args.sites:
        baseline_hashes = set(existing.loc[existing.site.eq(site), 'split_sha256'].dropna())
        multi_hashes = set(multi.loc[multi.site.eq(site), 'split_sha256'].dropna())
        if not baseline_hashes or not multi_hashes:
            raise ValueError(f'Site {site} missing prior or multi split metadata.')
        if len(baseline_hashes) != 1 or len(multi_hashes) != 1 or baseline_hashes != multi_hashes:
            raise ValueError(f'Site {site}: source/test split mismatch: {baseline_hashes} vs {multi_hashes}')

    candidates = []
    for (site, regime, w), group in multi.groupby(['site', 'regime', 'multi_weight']):
        if not validate_seeds(group, expected_seeds, (site, regime, w)):
            print(f'Skipping incomplete Multi config: site={site}, regime={regime}, w={w}, '
                  f'seeds={sorted(group.train_seed.tolist())}')
            continue
        candidates.append({
            'site': int(site), 'regime': str(regime), 'multi_weight': float(w),
            'n_seeds': len(group),
            'mean_source_val_acc': float(group.best_val.mean()),
            'sd_source_val_acc': float(group.best_val.std(ddof=1)),
        })
    candidates = pd.DataFrame(candidates)
    if candidates.empty:
        raise ValueError('No complete Multi configurations have all requested seeds.')
    missing_sites = set(args.sites) - set(candidates.site)
    if missing_sites:
        raise ValueError(f'No complete Multi sweep for sites {sorted(missing_sites)}.')

    # Selection rule: highest SOURCE validation mean, tie -> lower w, then strong.
    # Importantly, no OOD column enters this comparison.
    candidates['regime_tie_priority'] = candidates.regime.map({'strong': 0, 'weak': 1})
    ranked = candidates.sort_values(
        ['site', 'mean_source_val_acc', 'multi_weight', 'regime_tie_priority'],
        ascending=[True, False, True, True], kind='stable')
    selected = ranked.drop_duplicates(['site']).drop(columns='regime_tie_priority').sort_values('site').copy()
    selected['selection_rule'] = 'MAX mean source validation accuracy; tie -> smaller w then strong'

    # All OOD measurements happen *after* selection.
    multi_summary_rows = []
    for item in selected.itertuples(index=False):
        subset = multi[(multi.site == item.site) &
                       (multi.regime == item.regime) &
                       np.isclose(multi.multi_weight, item.multi_weight)]
        multi_summary_rows.append({
            'site': item.site, 'method': 'MixStyle + Multi',
            'mean_ood_bal_acc': float(subset.ood_bal_acc.mean()),
            'sd_ood_bal_acc': float(subset.ood_bal_acc.std(ddof=1)),
            'mean_in_bal_acc': float(subset.in_bal_acc.mean()),
            'sd_in_bal_acc': float(subset.in_bal_acc.std(ddof=1)),
            'n_seeds': len(subset),
        })
    multi_summary = pd.DataFrame(multi_summary_rows)

    old_rows = []
    for (site, method), group in existing.groupby(['site', 'method']):
        if method not in METHOD_LABELS:
            continue
        if not validate_seeds(group, expected_seeds, (site, method)):
            raise ValueError(f'Missing expected baseline seeds for {method} site {site}.')
        old_rows.append({
            'site': int(site), 'method': METHOD_LABELS[method],
            'mean_ood_bal_acc': float(group.ood_bal_acc.mean()),
            'sd_ood_bal_acc': float(group.ood_bal_acc.std(ddof=1)),
            'mean_in_bal_acc': float(group.in_bal_acc.mean()),
            'sd_in_bal_acc': float(group.in_bal_acc.std(ddof=1)),
            'n_seeds': len(group),
        })
    comparison = pd.concat([pd.DataFrame(old_rows), multi_summary], ignore_index=True)
    for site in args.sites:
        missing = set(METHOD_ORDER) - set(comparison.loc[comparison.site.eq(site), 'method'])
        if missing:
            raise ValueError(f'Incomplete methods for site {site}: {sorted(missing)}')

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sweep_csv = out / 'multi_source_validation_sweep.csv'
    selection_csv = out / 'multi_validation_selected_per_site.csv'
    comparison_csv = out / 'per_site_ood_comparison_six_methods.csv'
    candidates.drop(columns='regime_tie_priority').to_csv(sweep_csv, index=False)
    selected.to_csv(selection_csv, index=False)
    comparison.sort_values(['site', 'method']).to_csv(comparison_csv, index=False)

    # FIGURE 1: inspection/audit plot showing validation-based weight choice.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (site, regime), subset in candidates.groupby(['site', 'regime']):
        subset = subset.sort_values('multi_weight')
        label = f'Site {site}' if len(args.regimes) == 1 else f'Site {site} ({regime})'
        ax.errorbar(subset.multi_weight, 100*subset.mean_source_val_acc,
                    yerr=100*subset.sd_source_val_acc, marker='o', linewidth=1.4,
                    capsize=2, label=label)
    for item in selected.itertuples(index=False):
        ax.scatter([item.multi_weight], [100*item.mean_source_val_acc],
                   s=115, marker='*', edgecolors='black', linewidths=0.65, zorder=6)
    ax.set_xlabel('Multi-scale regularization weight w')
    ax.set_ylabel('Source validation accuracy (%)')
    ax.set_title('Multi-scale weight sweep: source validation only\nStars = selected site-specific configuration')
    ax.set_xticks(sorted(candidates.multi_weight.unique()))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'plot1_multi_validation_weight_sweep.png', dpi=200)
    fig.savefig(out / 'plot1_multi_validation_weight_sweep.pdf')
    plt.close(fig)

    # FIGURE 2: clean user-requested main figure (no w or regime in legend).
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for method in METHOD_ORDER:
        subset = comparison.loc[comparison.method.eq(method)].set_index('site').reindex(args.sites)
        ax.errorbar(args.sites, 100*subset.mean_ood_bal_acc,
                    yerr=100*subset.sd_ood_bal_acc,
                    marker='o', capsize=2.5, linewidth=1.6, label=method)
    ax.set_xlabel('Held-out clinical site')
    ax.set_ylabel('OOD balanced accuracy (%)')
    ax.set_xticks(args.sites, [f'Site {s}' for s in args.sites])
    ax.set_title('OOD balanced accuracy by site (five training seeds)\n'
                 'MixStyle + Multi: source-validation-selected configuration')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'plot2_ood_six_methods_with_multi.png', dpi=200)
    fig.savefig(out / 'plot2_ood_six_methods_with_multi.pdf')
    plt.close(fig)

    print('\nValidation-selected Multi hyperparameters (NOT selected on OOD):')
    print(selected[['site', 'regime', 'multi_weight', 'mean_source_val_acc']].to_string(index=False))
    print('\nSix-method per-site OOD comparison (mean ± SD):')
    print(comparison.sort_values(['site','method']).to_string(index=False,
         formatters={'mean_ood_bal_acc':lambda x:f'{x:.4f}',
                     'sd_ood_bal_acc':lambda x:f'{x:.4f}'}))
    print(f'\nCreated two plots and three CSV files under {out}/')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--existing', type=Path, default=Path('artifacts/results/multisite_runs/all_runs.csv'))
    parser.add_argument('--multi', type=Path, default=Path('artifacts/results/multiscale_runs/multi_all_runs.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/results/multiscale_runs'))
    parser.add_argument('--sites', type=int, nargs='+', default=[1,4,13,18,20])
    parser.add_argument('--regimes', choices=['strong','weak'], nargs='+', default=['strong'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    args = parser.parse_args()
    if len(set(args.sites)) != len(args.sites) or len(set(args.regimes)) != len(args.regimes):
        parser.error('Do not repeat sites or regimes.')
    if not args.existing.is_file() or not args.multi.is_file():
        parser.error('Missing input file. Run the Multi sweep first; expected:\n'
                     f' {args.existing}\n {args.multi}')
    create_figures(args)

if __name__ == '__main__':
    main()
