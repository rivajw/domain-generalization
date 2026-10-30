# Reconstructed MixStyle + Multi for BraTS2023

## Why this exists
The user's paper reports `MixStyle + Multi`, but the supplied training modules
and final notebook did not include the original multi-scale consistency training
function. This patch reconstructs the **described method**, not the missing
historical teammate implementation. Do **not** claim the new code reproduces the
historical reported scores without a new measured run.

### Paper-to-code mapping

- Architecture: pretrained ImageNet ResNet-18, only `layer2` and `fc` trainable.
- MixStyle after ResNet `layer2`, random donor permutation; disabled during evaluation.
- Clean forward pass: MixStyle OFF; no gradient, detach its features.
- Mixed forward pass: MixStyle ON; compute classification CE and a cosine
  difference relative to the clean representations.
- Capture four post-perturbation features: **after MixStyle at layer2**, after
  `layer3`, after `layer4`, and `avgpool` (512-D vector).
- Per-stage cosine loss: `mean_batch(1 - cosine(flatten(mixed), flatten(clean)))`.
- Multi objective: `CE + w * (cos_L2 + cos_L3 + cos_L4 + cos_pooled)`.
- Reported sweep: `w=0.25, 0.5, 1, 2, 4`.
- Reported regimes: strong MixStyle `(p=.75, alpha=.1)` and weak
  `(p=.5, alpha=.5)`.
- Training: Adam 1e-4, batch 128, 15 epochs maximum, patience=3, source
  validation accuracy checkpoint selection; fixed split seed=18662.
- Supports `site=18,1,4,13,20` and train seeds `0..4` to compare directly with
  existing 125-run `all_runs.csv`.

**Ambiguity in report:** The report does not specify whether intermediate maps
are flattened or spatially pooled before cosine, and whether stage terms are
summed or averaged. This reconstruction uses flattened spatial maps and a SUM
of four unweighted stage losses multiplied by per-stage weight `w`. These
choices are documented in result metadata and must be reported in the updated
paper. The original teammate implementation may differ.

## Install (on AWS)

The existing AWS project is `~/brats-dg/brats_dg`. It already contains
`run_dg_multisite.py`, `aws_paths.py`, the updated `dg` package, dataset folder,
and completed five-site runs.

1. Move the ZIP from your Mac to AWS (Mac terminal):

```bash
rsync -avhP -e "ssh -i ~/Downloads/YOUR_KEY.pem" \
  ~/Downloads/MixStyle_Multi_AWS_patch.zip \
  ubuntu@YOUR_EC2_IP:~/brats-dg/brats_dg/
```

2. From **normal AWS SSH shell (outside tmux)**:

```bash
cd ~/brats-dg/brats_dg
conda activate bratsdg
unzip -o MixStyle_Multi_AWS_patch.zip
python verify_multiscale.py
python run_mixstyle_multi.py --dry-run
```

Only `dg/multiscale.py` is added inside the existing `dg/` package. Neither
`dg/experiment.py` nor `dg/training.py`, and no old checkpoints, are replaced.

## Start one quick validation run (automatic tmux)

```bash
bash start_mixstyle_multi_tmux.sh --sites 18 --seeds 0 --regimes strong --weights 1
```

Inspect:

```bash
tmux ls
tmux attach -t <session_name_printed_by_launcher>
# Detach Ctrl+B then D
```

## Five-site reproducibility experiment (25 new trainings)

Run this after the one-run validation. The completed run is automatically skipped.

```bash
bash start_mixstyle_multi_tmux.sh \
  --sites 18 1 4 13 20 \
  --seeds 0 1 2 3 4 \
  --regimes strong --weights 1
```

To include **both** report regimes at `w=1` (50 trainings total, skipping
completed ones):

```bash
bash start_mixstyle_multi_tmux.sh \
  --sites 18 1 4 13 20 \
  --seeds 0 1 2 3 4 \
  --regimes strong weak --weights 1
```

To independently re-create the report's **weight sweep** (10 new runs for
site18, seed0, two regimes; skip existing runs):

```bash
bash start_mixstyle_multi_tmux.sh \
  --sites 18 --seeds 0 --regimes strong weak \
  --weights 0.25 0.5 1 2 4
```

**Publication caution:** Choosing the optimal weight by site18 OOD test
performance would be target leakage; use source-validation results for
hyperparameter selection, and freeze a choice before held-out evaluations.
Use original site18 sweep as exploratory analysis, not a pristine test.

## Output files

Per-run identifier includes `site`, `splitseed`, `trainseed`, `regime`, `p`,
`alpha` and `w`, e.g.
`DG-mixstyle_multi-site18-L2-split18662-seed0-strong-p0.75-a0.1-w1`.

- Checkpoints: `artifacts/checkpoints/best_<run-id>.pth`
- Logs: `artifacts/logs/multiscale_runs/<run-id>.log`
- History: `artifacts/results/multiscale_runs/<run-id>_history.json`
- Per-run metrics: `artifacts/results/multiscale_runs/<run-id>_metrics.json`
- Collated runs: `artifacts/results/multiscale_runs/multi_all_runs.csv`
- Per-site mean/std: `artifacts/results/multiscale_runs/multi_by_site_summary.csv`
- Macro-site summaries: `artifacts/results/multiscale_runs/multi_macro_summary.csv`
- Existing-model comparison (if original results CSV exists):
  `artifacts/results/multiscale_runs/with_existing_methods.csv`

Previously trained five-method experiments stay in
`artifacts/results/multisite_runs/all_runs.csv` and are never overwritten.

### Environment & important caveats

- These scripts use your existing Conda environment (`bratsdg`), and automatically
  launch a detached tmux session from your normal SSH shell.
- Run jobs sequentially on the single GPU; check `tmux ls` first to ensure
  you are not competing with another training session.
- `tmux` protects against SSH disconnects, **not** EC2 stop/restart.
- The runner will reuse existing checkpoints and skip completed results by name.
  `--force` retrains named experiments and overwrites only those new Multi files.
- The runner verifies patient-case split isolation and, when found, checks
  each site's split hash against the previously completed 125 runs.
- `python run_mixstyle_multi.py --help` lists every option.
