#!/usr/bin/env python3
"""Informational, read-only progress counter for the fixed-backbone 590-run plan.

Counts *completed* individual trial metrics JSON files, not checkpoints,
started jobs, training epochs or per-site summary rows. It never changes data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

SITES = (18, 1, 4, 13, 20)
VAL_BY_TARGET = {18: 1, 1: 18, 4: 1, 13: 1, 20: 1}
SEARCH_SEEDS = {0, 1}
FINAL_SEEDS = {100, 101, 102, 103, 104}
EXPECTED = {"style": 180, "kl": 40, "joint": 120, "multi": 100, "final": 150}
METHODS = {
    "style": {"mixstyle", "csu"},
    "kl": {"mixstyle_kl"},
    "joint": {"mixstyle_kl_mmd"},
    "multi": {"mixstyle_multi"},
    "final": {"erm", "mixstyle", "csu", "mixstyle_kl", "mixstyle_kl_mmd", "mixstyle_multi"},
}
P_VALUES = {0.25, 0.5, 0.75}
A_VALUES = {0.1, 0.3, 0.5}
KL_VALUES = {0.5, 1.0, 2.0, 3.0}
JOINT_KL_VALUES = {0.5, 1.0, 2.0}
JOINT_MMD_VALUES = {0.025, 0.05, 0.1, 0.25}
MULTI_W_VALUES = {0.25, 0.5, 1.0, 2.0, 4.0}


def valid_and_key(stage, row):
    try:
        site = int(row["site"])
        seed = int(row["train_seed"])
        method = row["method"]
        p = row["parameters"]
        if (site not in SITES or method not in METHODS[stage]
            or row.get("stage") != stage
            or int(row.get("val_site", -1)) != VAL_BY_TARGET[site]
            or int(row.get("val_case_cap", -1)) != 64
            or int(row.get("split_seed", -1)) != 18662
            or seed not in (FINAL_SEEDS if stage == "final" else SEARCH_SEEDS)
            or not row.get("name", "").startswith(f"SITEVAL-{stage}-{method}-target{site}-")
            or abs(float(p.get("lr", -1)) - 0.0001) > 1e-10
            or not 0 <= float(row["best_val_bal_acc"]) <= 1):
            return None
        if stage == "style":
            if float(p["p"]) not in P_VALUES or float(p["alpha"]) not in A_VALUES:
                return None
            config = (p["p"], p["alpha"])
        elif stage == "kl":
            if float(p["lambda_c"]) not in KL_VALUES:
                return None
            config = (p["lambda_c"],)
        elif stage == "joint":
            if float(p["lambda_c"]) not in JOINT_KL_VALUES or float(p["lambda_m"]) not in JOINT_MMD_VALUES:
                return None
            config = (p["lambda_c"], p["lambda_m"])
        elif stage == "multi":
            regime = p["regime"]
            if regime not in {"strong", "weak"} or float(p["w"]) not in MULTI_W_VALUES:
                return None
            if (float(p["p"]), float(p["alpha"])) != ({"strong": (0.75, 0.1), "weak": (0.5, 0.5)})[regime]:
                return None
            config = (regime, p["w"])
        else:
            if not 0 <= float(row["ood_bal_acc"]) <= 1:
                return None
            config = ()
        return (site, method, seed, *config)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def count_stage(root, stage):
    done = set()
    for path in (root / "artifacts" / "results" / f"siteholdout_{stage}").glob("*/*_metrics.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            key = valid_and_key(stage, row)
            if key is not None:
                done.add(key)
        except (OSError, json.JSONDecodeError):
            continue
    return min(len(done), EXPECTED[stage])


def count_original_multi(root):
    done = set()
    out = root / "artifacts" / "results" / "multiscale_runs"
    for path in out.glob("DG-mixstyle_multi-*_metrics.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            site = int(row["site"])
            seed = int(row["train_seed"])
            regime = row["regime"]
            w = float(row["multi_weight"])
            if (site in SITES and seed in {0, 1, 2, 3, 4}
                and regime in {"strong", "weak"} and w in MULTI_W_VALUES
                and int(row["split_seed"]) == 18662
                and 0 <= float(row["ood_bal_acc"]) <= 1):
                done.add((site, seed, regime, w))
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError):
            continue
    return min(len(done), 250)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--stage", default="waiting_multi",
                        choices=["waiting_multi", "style", "kl", "joint", "multi", "final", "other"])
    args = parser.parse_args()
    root = args.project_root.resolve()
    completed = {stage: count_stage(root, stage) for stage in EXPECTED}
    done = sum(completed.values())
    total = sum(EXPECTED.values())
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{now}] AUTOPILOT PROGRESS | current={args.stage} | site-holdout {done}/{total} ({done/total:.1%})", flush=True)
    print("    By stage: " + " | ".join(f"{s} {completed[s]}/{EXPECTED[s]} ({completed[s]/EXPECTED[s]:.0%})" for s in EXPECTED), flush=True)
    if args.stage == "waiting_multi":
        original_done = count_original_multi(root)
        print(f"    Original MixStyle+Multi sweep: {original_done}/250 ({original_done/250:.1%}) completed", flush=True)
    print("    Completed = metrics JSON saved; current running job/epoch not counted.", flush=True)


if __name__ == "__main__":
    main()
