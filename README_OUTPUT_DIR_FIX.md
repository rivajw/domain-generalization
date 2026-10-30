# Site-holdout output directory fix

Only replaces `run_dg_tuning_source_site.py` in the AWS project root.

Fix: create `artifacts/results/siteholdout_search/` and
`artifacts/results/siteholdout_final/` before saving CSV split manifests,
selection summaries, or final results.

Preserves all other training logic, the 590-run budget, fixed backbone,
and all current launcher/autopilot scripts. Existing metrics/checkpoints
are not deleted.

## Install

```bash
cd ~/brats-dg/brats_dg
conda activate bratsdg
unzip -o DG_siteval_output_dir_fix.zip
python -m py_compile run_dg_tuning_source_site.py
python run_dg_tuning_source_site.py --stage style --sites 18 --dry-run
```

Restart autopilot only if it is not currently running and you have verified
EC2 shutdown behavior is Stop, not Terminate:

```bash
bash start_dg_autopilot_tmux.sh --stop-method os --verified-shutdown-stop --multi-already-complete
```
