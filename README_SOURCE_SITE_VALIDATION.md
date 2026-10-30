# BraTS DG: fixed-backbone site-holdout tuning with two search seeds

This package **replaces the previous files with the same names** in `~/brats-dg/brats_dg/`:

- `run_dg_tuning_source_site.py` (modified default search seeds)
- `start_dg_source_site_tmux.sh` (modified: per-stage sessions)
- `start_dg_autopilot_tmux.sh` (modified launch checks)
- `dg_autopilot_supervisor.sh` (modified: stop after success **or** failure)

One new shared helper is included: `dg_siteval_pipeline.sh` (drives separate stage sessions, verifies success, captures errors). **No changes to `dg/` or the earlier MixStyle + Multi runner are required.** Existing `artifacts/` results and checkpoints are left in place. Existing runs are skipped only when their metrics and split agree; incompatible previously selected five-search-seed protocols trigger an explicit error rather than silently mixing protocols.

## Budget

| Stage | Configurations per target | Targets | Search seeds | Runs |
|---|---:|---:|---:|---:|
| style: MixStyle p×alpha 3×3 + CSU p×alpha 3×3 | 18 | 5 | 2 | 180 |
| kl: lambda_c 0.5, 1, 2, 3 | 4 | 5 | 2 | 40 |
| joint: lambda_c 0.5,1,2 × lambda_m 0.025,.05,.1,.25 | 12 | 5 | 2 | 120 |
| multi: two p/alpha regimes × w .25,.5,1,2,4 | 10 | 5 | 2 | 100 |
| final: six selected models × final seeds 100..104 | 6 | 5 | 5 | 150 |
| **Total** | | | | **590** |

Search seeds: `0,1`; final independent seeds: `100,101,102,103,104`. Fixed split seed: `18662`. Five true target sites: `18,1,4,13,20`. One *entire* source validation site is held out for each target: `{18:1, 1:18, 4:1, 13:1, 20:1}`. Validate up to 64 deterministic cases from that site (all its other cases are entirely excluded). Source-validation **balanced accuracy** selects checkpoint epochs and hyperparameters; real OOD site is evaluated only in final stage. Selected hyperparameters are site-specific. The ImageNet-pretrained ResNet-18 L2+FC backbone is trained using a fixed Adam learning rate of 1e-4, matching the existing project implementation. This is **not** claimed as a verbatim reproduction of either original paper’s optimizer settings. The plain ResNet baseline is still trained for the five final seeds on all five target sites (25 runs). No ResNet hyperparameter selection is performed. Training layers, losses, DG-specific hyperparameter grids, naming and result layout are unchanged.

## Installation / verification

From a **regular SSH terminal**, outside tmux:

```bash
cd ~/brats-dg/brats_dg
conda activate bratsdg
unzip -o DG_source_site_validation_reduced_AWS.zip
python -c 'import torch; from dg.training import train_epoch_kl_mmd; from dg.multiscale import train_epoch_multiscale; print("CUDA", torch.cuda.is_available())'
python run_dg_tuning_source_site.py --stage style --sites 18 --dry-run
```

Expected dry-run output includes `search seeds: [0, 1]` and `TARGET=18 VALIDATION_SITE=1`.

## How tmux sessions work now

- `dg_autopilot`: the **supervisor** that waits for your currently running `multi_*` experiment, schedules stage sessions, and stops EC2 when safe.
- `dg_siteval_style`, `dg_siteval_kl`, `dg_siteval_joint`, `dg_siteval_multi`, `dg_siteval_final`: **separate sessions started one at a time**, each with its own durable log and numeric process exit marker.
- Completed sessions are cleaned up by the manager; their logs, metrics and checkpoints remain. If an active session unexpectedly disappears without writing an exit marker, that stage is marked failed and later stages do not launch.
- `bash start_dg_source_site_tmux.sh all` uses a `dg_siteval_all` **manager session** plus the five sequential stage sessions. It does **not** monitor the old Multi experiment or stop the instance. Use the autopilot for waiting and automatic stopping.

### Recommended: wait for existing Multi, then run and auto-stop

1. In EC2 console, verify **Actions → Instance settings → Change shutdown behavior = Stop** (NOT Terminate).
2. While the earlier `multi_*` tmux session is still running, check prerequisites:

```bash
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop --dry-run
```

3. Start the supervisor (in ordinary SSH terminal, after `conda activate bratsdg`):

```bash
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop
```

Alternative when EC2 instance role has permission for `ec2:StopInstances` (and AWS CLI is installed):

```bash
bash start_dg_autopilot_tmux.sh --stop-method aws --dry-run
bash start_dg_autopilot_tmux.sh --stop-method aws
```

Check `tmux ls`; view `tmux attach -t dg_autopilot` (detach `Ctrl+B`, `D`). Supervisor log/state: `artifacts/logs/dg_autopilot/`.

**Important:** If you previously killed `dg_autopilot`, it is not running; it must be restarted if you want automatic EC2 stopping. Only one supervisor or full-study manager should be running at a time.

### Manual, stage-by-stage

```bash
bash start_dg_source_site_tmux.sh style
bash start_dg_source_site_tmux.sh kl
bash start_dg_source_site_tmux.sh joint
bash start_dg_source_site_tmux.sh multi
bash start_dg_source_site_tmux.sh final
```

Or use `bash start_dg_source_site_tmux.sh all` to run them sequentially without auto EC2 stop; don't run this if the previous Multi job is still active on a single GPU.

## Output folders and failure handling

Unchanged established output directories:

```text
artifacts/checkpoints/siteholdout_<stage>/<method>/best_*.pth
artifacts/results/siteholdout_<stage>/<method>/*_metrics.json
artifacts/logs/siteholdout_<stage>/<method>/*.log
artifacts/results/siteholdout_search/selected_hparams.csv
artifacts/results/siteholdout_search/selected_hparams.json
artifacts/results/siteholdout_final/all_final_runs.csv
artifacts/results/siteholdout_final/per_site_comparison.csv
artifacts/results/siteholdout_final/per_site_ood_tuned.png
```

New orchestration files are stored under existing logging paths:

```text
artifacts/logs/siteholdout_launcher/dg_siteval_<stage>_<timestamp>.log
artifacts/logs/siteholdout_launcher/dg_siteval_<stage>_<timestamp>.exit
artifacts/logs/siteholdout_launcher/errors/failure_*_<stage>_*.txt
artifacts/logs/dg_autopilot/errors/autopilot_error_*.txt
artifacts/logs/dg_autopilot/autopilot_*.state
artifacts/logs/dg_autopilot/autopilot_*.log
```

An error file contains the **failed stage**, the **tmux session name**, a failure reason/exit code, and the **last 200 lines of the stage terminal output**. The full per-stage log remains on disk. The supervisor stops scheduling subsequent stages and requests a safe EC2 STOP **even after failure**. A failed run with no metrics JSON can be rerun later; previously completed matching runs are skipped.

### Safety limitations: read before unattended use

- **Shutdown behavior MUST be Stop, NOT Terminate** when using `--stop-method os`; otherwise `shutdown -h now` may terminate EC2. The script requires an explicit `--verified-shutdown-stop` acknowledgement. `--stop-method aws` uses the instance IAM role and checks StopInstances permission before launch.
- To avoid killing unrelated compute work, auto-stop **waits for all other tmux sessions and GPU compute jobs to finish**. Thus it can remain running if an unrelated tmux session persists or `nvidia-smi` fails. This cannot guarantee zero idle charges. Configure an independent AWS cost alarm/scheduled stop as a fail-safe for unexpected host or supervisor failures.
- An EC2 host failure, disk full, lost power, killed supervisor, or IAM/OS shutdown misconfiguration can prevent auto-stop. Verify EC2 is actually `stopped` afterward. EBS storage charges persist after stopping.
- `--stop-method os` relies on EC2 instance shutdown behavior. It does not `terminate-instances` or delete EBS disks.
- Logging and atomic metrics saving minimize loss, but an interrupted *individual* trial may need to restart from epoch 1; optimizer-state resume is not supported.
- This two-search-seed setup does not have the same protocol as the earlier five-search-seed plan. Never merge both parameter-selection protocols without disclosure.

## Fixed-backbone protocol

The ResNet-18 backbone architecture, trainable layers (L2+FC) and optimizer/learning rate are held constant across the ERM and all DG methods. The Adam LR is fixed at `1e-4`, matching the current `dg.experiment.build_model()` default. No ResNet learning-rate search is performed. The complete study is **590 runs** = 440 search runs + 150 final runs, with search seeds `0,1` and evaluation seeds `100..104`. The ERM baseline is evaluated in `final`, rather than selected from a hyperparameter grid.

The five sequential stage sessions are: `dg_siteval_style`, `dg_siteval_kl`, `dg_siteval_joint`, `dg_siteval_multi`, and `dg_siteval_final`. The autopilot still monitors the existing `multi_*` tmux and attempts to stop EC2 safely on completion or logged failure.

If an older learning-rate-tuning protocol has *already written* `artifacts/results/siteholdout_search/protocol_site*.json` files, the fixed-LR code deliberately refuses to silently mix the two protocols. Back up that protocol/results directory and start a clean fixed-LR study before running.
