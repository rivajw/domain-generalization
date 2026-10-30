"""Shared AWS/Colab-independent data setup for the BraTS DG scripts.

Only changes to notebook behavior: the files in index.csv are remapped in memory
from their old Google Drive paths to the --data-root location on this machine.
No source index file is overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.getenv("BRATS_DATA_ROOT", str(PROJECT_ROOT / "brats23_dg_png"))).expanduser()


def add_common_args(parser):
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing index.csv and site_xx/*.npz (default: project/brats23_dg_png)")
    parser.add_argument("--index", type=Path, default=None, help="Optional custom index.csv path")
    parser.add_argument("--site", type=int, default=18, help="Held-out test site (default: 18)")
    parser.add_argument("--seed", type=int, default=18662, help="Experiment seed (default: 18662)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--allow-cpu", action="store_true", help="Opt into CPU training")
    parser.add_argument("--dry-run", action="store_true", help="Verify files/splits but do not train")


def find_index(data_root: Path, explicit_index: Path | None) -> Path:
    candidates = ([explicit_index] if explicit_index else [
        data_root / "index.csv", PROJECT_ROOT / "brats23_dg_png" / "index.csv",
        PROJECT_ROOT / "brats23_npz_dg" / "index.csv",
    ])
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError("index.csv not found; specify --data-root or --index. Checked: " +
                            ", ".join(map(str, candidates)))


def resolve_npz(raw_path: str, data_root: Path, index_parent: Path) -> Path | None:
    original = Path(str(raw_path).strip()).expanduser()
    if original.is_file():
        return original.resolve()
    raw = str(raw_path).replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    candidates = []
    # The original notebook used: /content/drive/MyDrive/18662/project/brats23_dg_png/site_xx/CASE.npz
    for dirname in (data_root.name, "brats23_dg_png", "brats23_npz_dg"):
        if dirname in parts:
            at = len(parts) - 1 - parts[::-1].index(dirname)
            candidates.append(data_root.joinpath(*parts[at + 1:]))
    # Accept paths with the site_xx directory even if the original root name differs.
    for i, part in enumerate(parts):
        if part.startswith("site_"):
            candidates.append(data_root.joinpath(*parts[i:]))
            break
    candidates.extend([index_parent / raw, data_root / raw])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_index(data_root: Path, index_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    root = data_root.expanduser().resolve()
    ix = find_index(root, index_path)
    df = pd.read_csv(ix)
    needed = {"case_id", "site", "npz_path", "num_slices", "num_tumor", "num_normal"}
    if not needed.issubset(df.columns):
        raise ValueError(f"index.csv is missing columns {sorted(needed - set(df.columns))}")
    if df.empty:
        raise ValueError(f"Empty index: {ix}")
    mapped = [resolve_npz(p, root, ix.parent) for p in df["npz_path"]]
    missing = [str(src) for src, dest in zip(df["npz_path"], mapped) if dest is None]
    if missing:
        preview = "\n".join(" - " + m for m in missing[:5])
        raise FileNotFoundError(
            f"Could not locate {len(missing)}/{len(mapped)} NPZ files under {root}.\n"
            f"Examples from index.csv:\n{preview}\n"
            "Make sure you transferred the full brats23_dg_png folder, including its site_* subfolders. "
            "If its location differs, supply --data-root /absolute/path/to/brats23_dg_png."
        )
    df = df.copy()
    df["npz_path"] = [str(x) for x in mapped]
    df["site"] = df["site"].astype(int)
    print(f"Index: {ix} | cases={df.case_id.nunique()} | sites={df.site.nunique()} | verified NPZ={len(df)}", flush=True)
    print(f"Remapped sample: {df.iloc[0].npz_path}", flush=True)
    return df, ix


def create_data_split(index_df: pd.DataFrame, site: int, seed: int):
    from dg.splits import build_main_split
    from dg.experiment import DataSplitConfig

    split = build_main_split(index_df, test_sites=[site], seed=seed)
    if not split["out_domain_test_set"]:
        raise ValueError(f"No OOD cases for site {site} in index")
    data_cfg = DataSplitConfig(
        name="default",
        train_cases=frozenset(split["train_set"]),
        validation_cases=frozenset(split["val_set"]),
        test_cases=frozenset(split["in_domain_test_set"]),
        out_domain_test_cases=frozenset(split["out_domain_test_set"]),
    )
    keys = ["train_cases", "validation_cases", "test_cases", "out_domain_test_cases"]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if getattr(data_cfg, a) & getattr(data_cfg, b):
                raise AssertionError(f"Case leakage: {a} overlaps with {b}")
    print("Case counts (train / val / in-domain / OOD): " +
          " / ".join(str(len(getattr(data_cfg, x))) for x in keys), flush=True)
    print("Held-out site(s):", sorted(split["out_domain_test_df"].site.unique()), flush=True)
    return data_cfg, split


def save_split_manifest(split: dict, site: int, seed: int) -> Path:
    from dg import config
    rows = []
    for label, key in (("train", "train_df"), ("validation", "val_df"),
                       ("in_domain_test", "in_domain_test_df"), ("out_domain_test", "out_domain_test_df")):
        block = split[key][["case_id", "site"]].drop_duplicates().copy()
        block["split"] = label
        rows.append(block)
    path = Path(config.RESULTS_DIR) / f"split_site{site}_seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    print("Split manifest:", path, flush=True)
    return path


def require_training_device(allow_cpu: bool) -> None:
    from dg import runtime
    print(f"Device={runtime.DEVICE}, AMP={runtime.USE_AMP}", flush=True)
    if runtime.DEVICE.type != "cuda" and not allow_cpu:
        raise RuntimeError("CUDA GPU not detected. Use an AWS GPU instance with a CUDA-enabled PyTorch "
                           "installation; or pass --allow-cpu if intentional.")
