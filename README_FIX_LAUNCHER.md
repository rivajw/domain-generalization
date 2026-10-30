# DG launcher fix (2026-10-28)

The original 250-run MixStyle+Multi study was successful, but the first site-holdout tuning stage failed at its CLI invocation with `run_dg_tuning_source_site.py: error: unrecognized arguments:` (empty argument). The root cause was `printf -v PARAM_Q '%q ' "$@"` when the stage received no extra arguments. In Bash, this creates a quoted empty string (`'' `), which was inadvertently included as a positional argument.

This patch changes exactly these three files:

- `start_dg_source_site_tmux.sh`: sets `PARAM_Q=''` when there are no extra args; formats only nonempty optional argument vectors. No optimizer, splits, hyperparameter grids, or result directory changes.
- `dg_siteval_pipeline.sh`: maps internal progress phases such as `preflight` and `final_integrity` to `other`, an accepted value for the read-only progress reporter. No training logic changes.
- `start_dg_autopilot_tmux.sh`: adds an **opt-in** `--multi-already-complete` option for recovery when the original Multi sweep has completed and EC2 reboot removed its tmux sessions. Requires verifying all 250 unique saved Multi results. It does not silently treat a missing tmux session as success.

The project remains the fixed-backbone 590-run protocol: style (180) -> kl (40) -> joint (120) -> multi (100) -> final (150). Search seeds 0,1; final seeds 100..104. Each stage uses a distinct detached tmux session; outputs remain under `artifacts/`.

## Install

Copy `DG_siteval_launcher_recovery_patch.zip` to `~/brats-dg/brats_dg/` and run:

```bash
cd ~/brats-dg/brats_dg
conda activate bratsdg
unzip -o DG_siteval_launcher_recovery_patch.zip
for f in start_dg_source_site_tmux.sh dg_siteval_pipeline.sh start_dg_autopilot_tmux.sh; do bash -n "$f" || exit 1; done
python run_dg_tuning_source_site.py --stage style --sites 18 --dry-run
```

## Resume the full tuning automatically, with EC2 stopping

After checking EC2 > Instance settings > Change shutdown behavior is **Stop**, not Terminate:

```bash
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop --multi-already-complete --dry-run
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop --multi-already-complete
```

This command is **only** for the special case where the original 250-run Multi sweep is fully finished and no `multi_*` tmux session remains. Otherwise, omit the flag and use the original wait-for-running-Multi workflow.

**Caution:** This autostop supervisor requests EC2 stop both on successful completion *and* a detected stage failure. Verify you can access logs/checkpoints on EBS. Stopping the instance removes live tmux sessions and can lead to EC2 capacity-related restart delays. EBS storage charges may continue while compute is stopped.

Track logs:

```bash
tmux ls
tmux attach -t dg_autopilot
ls -lt artifacts/logs/dg_autopilot/
ls -lt artifacts/logs/siteholdout_launcher/
```

To detach from tmux: Ctrl+B, then D.

## Validation performed

- Bash syntax checks passed for all three changed scripts.
- A mocked stage launch with **zero** extra CLI options produced exactly `-u <runner> --stage style`, with no empty positional argument; exit status recorded successfully.
- A mocked resume with 250 distinct, valid Multi metric records passed the completion guard; removing one metric caused the guard to refuse recovery.
- This was not executed on a real AWS EC2 GPU.
