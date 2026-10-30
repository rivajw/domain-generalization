# BraTS DG reproducibility patch: five sites, five seeds, CSU audit

## What this patch is

A **patch** for the existing AWS project at `~/brats-dg/brats_dg`.

It adds a controlled, fixed-split multi-site/multi-seed runner and training support
for MixStyle + KL + class-conditional MMD. It does **not** include the BraTS data
or replace other existing project files.

## What was audited against CSU WACV 2024

Paper: Zhang et al., *Domain Generalization With Correlated Style Uncertainty*, WACV 2024:
https://arxiv.org/html/2212.09950
Authors' source: https://github.com/freshman97/CSU

The independent `dg/csu.py` matches the published **algorithm equations**:

- Eq. (1–2): channel-wise mean and std of `[B,C,H,W]` features.
- Eq. (3–4): mini-batch covariance with divisor B (not B-1).
- Eq. (5–8), (11–13): symmetric eigendecomposition and `Q sqrt(Lambda) Q^T` transforms independent Gaussian noise to correlated noise.
- Eq. (14–16): `new_mean = mu + lambda*delta_mu`, `new_std = sigma + lambda*delta_sigma`; apply to standardized feature.
- Beta(alpha,alpha) used for lambda; activation probability p.

`verify_csu_math.py` numerically checks covariance, symmetric square root,
empirical correlated sampling, gradient flow, and inference identity.

**Not established:** bit-for-bit identity with the author's implementation. The
independent implementation was tested against the published math; we did not
execute their published repository in this environment. Cite this as
"CSU adapted from the published formulation", not "official CSU reproduced".

Data/model adaptation: 2D T2-FLAIR slices from site-labeled BraTS NPZs,
3 repeated intensity channels, ImageNet ResNet18, layer2 insertion, partial L2
and FC fine-tuning. These change the *data task, backbone pipeline, training*
but not the CSU feature-statistic transformation.

Implementation choices: float32 eigendecomposition under AMP, epsilon in std,
style statistics detached, batch-level p activation, independently sampled
per-image Beta coefficient, and no perturbation at eval. Those can change
numerics/stochastic paths versus any differing author implementation, so
we cannot assert experimental equivalence.

## Five sites + five training seeds

Sites: `18 1 4 13 20` (site18 hardest gap + large site1 + challenging site13
+ near-neutral site4 + easier site20). The sites are **predeclared**, not
reselected after these experiments.

Training seeds: `0 1 2 3 4` (five distinct fixed RNG seeds; there is no canonical
five-seed list mandated by MixStyle or CSU). Fixed **split seed** `18662` is used
for all five training seeds **within a given site**. Each site has a different,
case-disjoint LOSO split because the target hospital changes.

Methods:
- `erm`: ResNet-18 L2 baseline
- `mixstyle`: MixStyle L2 p=0.75 alpha=0.1
- `csu`: CSU L2 p=0.75 alpha=0.1
- `mixstyle_kl`: MixStyle L2 + KL weight 1.0
- `mixstyle_kl_mmd`: MixStyle L2 + KL weight 2.0 + class-conditional MMD weight 0.1

All use the original `dg` training configuration (Adam 1e-4, batch128, <=15 epochs,
patience3, ImageNet initialization). The code retains the existing source-validation
accuracy-based checkpoint selection.

**Combined-loss implementation note:** Original AWS `dg/experiment.py` rejected
KL+MMD together. This patch adds a combined clean-teacher / mixed-student
training path using one mixed pass for CE, KL, and class-conditional MMD. This is
NEW code and must be treated as a new reproducibility run; you must NOT claim
that it exactly reproduced the original 0.9048 historical result.

**KL orientation discrepancy:** The original code computes
`F.kl_div(log_softmax(mixed), softmax(clean), reduction='batchmean')`, which is
`KL(p_clean || p_mixed)` mathematically. Some manuscript text calls this
`KL(p_mixed || p_clean)`. The patch intentionally preserves the existing
implemented direction for comparability. Correct the manuscript or undertake a
separate ablation; DO NOT silently change it when interpreting previous runs.

## Install on AWS

Transfer `dg_multisite_repro_patch.zip` into:

`~/brats-dg/brats_dg/`

In your **normal SSH terminal, outside tmux**:

```bash
cd ~/brats-dg/brats_dg
# Optional backups of files about to change:
cp dg/experiment.py dg/experiment.py.backup
cp dg/training.py dg/training.py.backup
unzip -o dg_multisite_repro_patch.zip
conda activate bratsdg
python verify_csu_math.py
python run_dg_multisite.py --dry-run
```

`dg/csu.py` is included unchanged from the previously supplied CSU patch to
make its audited source explicit; any existing CSU checkpoint is left alone.

## Run inside tmux automatically

For a first trial with just CSU and MixStyle on site18:

```bash
bash start_dg_multisite_tmux.sh --sites 18 --methods csu mixstyle --seeds 0 1 2 3 4
```

If you want to run the entire predeclared study **instead**:

```bash
bash start_dg_multisite_tmux.sh
```

NOTE: The full default grid is **5 sites x 5 seeds x 5 methods = 125 trainings**.
It may be expensive. Each method is trained serially on the one GPU.

To monitor:

```bash
tmux ls
tmux attach -t dg_5site_5seed
# Ctrl+B, then D to detach
ls -lt artifacts/logs/multisite_runs/ | head
```

To run another batch after the previous tmux session ends, run the launcher
again with different --sites/--methods. Completed runs are skipped. If the
previous session remains open, detach and stop/end it before restarting the
launcher (the name is intentionally fixed to avoid simultaneous GPU jobs).

## Files and resumption

- Individual checkpoints: `artifacts/checkpoints/best_DG-METHOD-siteSITE-L2-split18662-seedSEED*.pth`
- Original history JSON: `artifacts/results/DG-METHOD-siteSITE-L2-split18662-seedSEED*.json`
- Per-run metrics: `artifacts/results/multisite_runs/DG-*_metrics.json`
- Per-run logs: `artifacts/logs/multisite_runs/DG-*.log`
- Per-site fixed manifests: `artifacts/results/multisite_runs/split_siteSITE_splitseed18662.csv`
- Summary (updated after each run):
  - `artifacts/results/multisite_runs/all_runs.csv`
  - `artifacts/results/multisite_runs/by_site_summary.csv` (mean+SD per method and site)
  - `artifacts/results/multisite_runs/by_method_macro_summary.csv` (equal-weight mean over sites)

The existing site18 CSU checkpoints named `best_CSU-L2-...` are NOT used by
this new fixed-split/seed protocol because filenames now encode site,
split seed, and train seed.

The runner skips completed metrics and re-evaluates a saved checkpoint if it
exists without an accompanying metrics file. It rejects a changed split
manifest. `--force` intentionally retrains a named run and overwrites its
checkpoint, history and metrics.

## Publication protocol caveat

You previously chose site18 after examining baseline OOD gaps, and some
hyperparameter settings were selected on the site18 OOD test. Hence site18
is an **exploratory/stress-test domain**, not a genuinely untouched final test.
Other sites now used to assess robustness must not be used to choose settings.
Report the fact that the method configurations were frozen ahead of this run.

Caution: Slice-level observations within one patient are correlated; future
confidence intervals should be patient-clustered and evaluations should be
case-disjoint. Five training seeds characterize optimization variability for
a fixed split; they do not replace case-level statistical uncertainty.
