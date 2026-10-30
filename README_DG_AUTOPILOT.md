# Updated DG autopilot: sequential stage-specific tmux and stop on error

This README accompanies `DG_source_site_validation_reduced_AWS.zip`.
See `README_SOURCE_SITE_VALIDATION.md` for complete instructions and safeguards.

The prior autopilot kept EC2 running after a tuning failure. **This version records the failing stage/session and the terminal traceback, then requests an EC2 STOP once no unrelated tmux sessions or GPU compute jobs remain.** The six tuning stages run in six separate, sequential tmux sessions. A supervisor `dg_autopilot` waits for the already-running `multi_*` job and orchestrates the process. The same old script names are overwritten.

Launch after confirming EC2 instance-initiated shutdown behavior is **Stop**:

```bash
cd ~/brats-dg/brats_dg
conda activate bratsdg
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop --dry-run
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop
```

An EC2 IAM-role-backed AWS stop method is also supported: `--stop-method aws`. Check the logs under `artifacts/logs/dg_autopilot/`. **Please verify in AWS Console that the instance is actually stopped**; no in-instance script can guarantee shutdown after a host or permission failure. An external AWS budget alarm is a useful independent fail-safe.

**Fixed backbone update:** the pipeline skips ResNet LR search. It launches five stages (style, kl, joint, multi, final), and the final stage still includes the fixed ResNet-18 baseline. The `run_dg_tuning_source_site.py` runner remains the same filename.
