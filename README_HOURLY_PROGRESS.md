# Hourly progress reports for DG autopilot (590-run fixed-backbone plan)

This is a **monitoring-only patch** for `DG_source_site_fixed_backbone_AWS.zip`. It retains the existing training runner, stage scheduling, run count, success/failure rules, logs, and EC2 stop behavior.

## Install

Copy `DG_autopilot_hourly_progress_patch.zip` to `~/brats-dg/brats_dg/` on your EC2 instance:

```bash
cd ~/brats-dg/brats_dg
conda activate bratsdg
unzip -o DG_autopilot_hourly_progress_patch.zip
bash -n dg_autopilot_supervisor.sh dg_siteval_pipeline.sh
python dg_autopilot_progress.py --stage waiting_multi
```

The patch replaces only these two existing files and adds the counter:

- `dg_autopilot_supervisor.sh` — reports immediately and every 3600 seconds while waiting for the original Multi job.
- `dg_siteval_pipeline.sh` — reports immediately, at least once every 3600 seconds during tuning, and on every stage completion.
- `dg_autopilot_progress.py` — counts valid completed `*_metrics.json` trials (read-only) by study stage.

Expected tuning counts: `style 180`, `kl 40`, `joint 120`, `multi 100`, `final 150`; total **590**. The original MixStyle+Multi sweep, if running, is separately displayed as `N/250` and does **not** contribute to 590.

Example displayed output:

```text
[2026-10-28 12:00:00 EDT] AUTOPILOT PROGRESS | current=joint | site-holdout 267/590 (45.3%)
    By stage: style 180/180 (100%) | kl 40/40 (100%) | joint 47/120 (39%) | multi 0/100 (0%) | final 0/150 (0%)
    Completed = metrics JSON saved; current running job/epoch not counted.
```

Progress also appears in the `dg_autopilot` tmux pane and the autopilot log under `artifacts/logs/dg_autopilot/`.

**Important:** Overwriting a shell script **does not change a shell process that is already running**. If the supervisor is **still WAITING_FOR_MULTI**, you can stop only that supervisor (`tmux kill-session -t dg_autopilot`), install the patch, and relaunch the same command **while the original `multi_*` tmux session remains active**:

```bash
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop
```

First verify instance-initiated shutdown is `Stop`, not `Terminate`, as with the previous package. **Do not restart the autopilot if site-holdout tuning has already started**: it may conflict with an active `dg_siteval_*` session. In that situation use the manual progress command to check status without touching any running job:

```bash
python dg_autopilot_progress.py --stage joint
```

No automatic changes to AWS instance state are made by this reporting script. The autopilot's existing EC2 stop safeguards still apply.
