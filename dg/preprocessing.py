"""BraTS 2023 GLI preprocessing: ZIP -> per-case NPZ dataset.

Pipeline
--------
1. Read the BraTS 2023 GLI TrainingData ZIP
2. Assign each case to a site (hospital) via the mapping .xlsx
3. Convert 3D volumes to 2D axial slices
4. Label each slice tumor/normal from the ``*-seg.nii.gz`` mask
5. Cap slices per case to keep dataset size small
6. Save one compressed ``.npz`` per case under ``site_xx/``
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from . import config


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------
def download_from_synapse(zip_path: str = config.ZIP_PATH, token: str = config.TOKEN) -> str:
    """Download BraTS ZIP from Synapse if it isn't already on disk."""
    import synapseclient

    zip_path = str(Path(zip_path).expanduser())
    if Path(zip_path).exists():
        print(f"Using local ZIP_PATH: {zip_path}")
        return zip_path

    print("Dataset zip missing, downloading from Synapse...")
    syn = synapseclient.Synapse()
    syn.login(authToken=token)
    entity = syn.get(entity="syn51514132", downloadLocation="./")
    return entity.path


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def parse_case_id(path_in_zip: str) -> str | None:
    m = re.search(r"(BraTS-GLI-\d{5}-\d{3})", path_in_zip)
    return m.group(1) if m else None


def load_nii_from_zip(z: zipfile.ZipFile, member: str) -> np.ndarray:
    """Read .nii.gz bytes from zip -> temp file -> nibabel -> numpy array."""
    data = z.read(member)
    fd, tmp_path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        return nib.load(tmp_path).get_fdata().astype(np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def zscore_nonzero(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mask = vol != 0
    if mask.sum() < 100:
        return (vol - vol.mean()) / (vol.std() + eps)
    vals = vol[mask]
    mu, sigma = vals.mean(), vals.std() + eps
    out = vol.copy()
    out[mask] = (out[mask] - mu) / sigma
    out[~mask] = 0.0
    return out


def slice_to_uint8(img2d: np.ndarray) -> np.ndarray:
    """Robust percentile scaling to uint8."""
    x = img2d.astype(np.float32)
    nz = x[x != 0]
    if nz.size < 10:
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(nz, [1, 99])
    if (hi - lo) < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return (x * 255.0).astype(np.uint8)


def resize_uint8(img_u8: np.ndarray, out_hw=(224, 224)) -> np.ndarray:
    pil = Image.fromarray(img_u8, mode="L")
    pil = pil.resize(out_hw, resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


def choose_evenly(indices: list[int], max_keep: int) -> list[int]:
    """Evenly subsample a list to length max_keep (deterministic)."""
    if len(indices) <= max_keep:
        return indices
    idxs = np.linspace(0, len(indices) - 1, num=max_keep).astype(int)
    return [indices[i] for i in idxs]


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_zip_to_npz(
    zip_path: str = config.ZIP_PATH,
    mapping_xlsx: str = config.MAPPING_XLSX,
    out_root: str = config.OUT_ROOT,
    modality: str = config.MODALITY,
    axis: int = config.AXIS,
    min_brain_pixels: int = config.MIN_BRAIN_PIXELS,
    tumor_pixel_threshold: int = config.TUMOR_PIXEL_THRESHOLD,
    max_tumor_slices: int = config.MAX_TUMOR_SLICES,
    max_normal_slices: int = config.MAX_NORMAL_SLICES,
    out_size: tuple[int, int] = config.OUT_SIZE,
) -> pd.DataFrame:
    """Convert the BraTS ZIP into a compact per-case NPZ dataset.

    If ``out_root`` is already populated, the pipeline is skipped and the
    existing ``index.csv`` is loaded instead.
    """
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(out_dir.iterdir()):
        print(f"OUT_ROOT already exists, skipping unzip/processing: {out_dir}")
        index_csv = out_dir / "index.csv"
        if index_csv.exists():
            return pd.read_csv(index_csv)
        return pd.DataFrame()

    # Load case -> site mapping
    df_map = pd.read_excel(mapping_xlsx)
    case_col = "BraTS2023"
    site_col = "Site No (represents the originating institution)"
    df_map = df_map[[case_col, site_col]].dropna()
    df_map[site_col] = df_map[site_col].astype(int)
    case_to_site = dict(zip(df_map[case_col].astype(str), df_map[site_col]))
    print("Mapping entries:", len(case_to_site))

    index_rows = []
    skipped_no_site = 0
    skipped_missing = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        case_members: dict[str, list[str]] = {}
        for n in names:
            cid = parse_case_id(n)
            if cid:
                case_members.setdefault(cid, []).append(n)
        print("Cases in zip:", len(case_members))

        for k, (case_id, members) in enumerate(case_members.items(), start=1):
            site = case_to_site.get(case_id, None)
            if site is None:
                skipped_no_site += 1
                continue

            img_path = next((m for m in members if m.endswith(f"-{modality}.nii.gz")), None)
            seg_path = next((m for m in members if m.endswith("-seg.nii.gz")), None)
            if img_path is None or seg_path is None:
                skipped_missing += 1
                continue

            img_vol = zscore_nonzero(load_nii_from_zip(z, img_path))
            seg_vol = load_nii_from_zip(z, seg_path)
            if img_vol.shape != seg_vol.shape:
                skipped_missing += 1
                continue

            D = img_vol.shape[axis]
            tumor_idxs, normal_idxs = [], []
            for s in range(D):
                img_slice = img_vol[:, :, s]
                seg_slice = seg_vol[:, :, s]
                if (img_slice != 0).sum() < min_brain_pixels:
                    continue
                tumor_pixels = int((seg_slice > 0).sum())
                if tumor_pixels >= tumor_pixel_threshold:
                    tumor_idxs.append(s)
                else:
                    normal_idxs.append(s)

            tumor_keep = choose_evenly(tumor_idxs, max_tumor_slices)
            normal_keep = choose_evenly(normal_idxs, max_normal_slices)

            slices, labels = [], []
            for s in tumor_keep:
                slices.append(resize_uint8(slice_to_uint8(img_vol[:, :, s]), out_size))
                labels.append(1)
            for s in normal_keep:
                slices.append(resize_uint8(slice_to_uint8(img_vol[:, :, s]), out_size))
                labels.append(0)

            if len(slices) == 0:
                continue

            x = np.stack(slices, axis=0).astype(np.uint8)  # [N,224,224]
            y = np.array(labels, dtype=np.uint8)            # [N]

            site_dir = out_dir / f"site_{site:02d}"
            site_dir.mkdir(parents=True, exist_ok=True)
            out_npz = site_dir / f"{case_id}.npz"
            np.savez_compressed(
                out_npz,
                images=x,
                labels=y,
                case_id=np.array(case_id),
                site=np.array(site, dtype=np.int32),
                modality=np.array(modality),
            )

            index_rows.append(
                {
                    "case_id": case_id,
                    "site": site,
                    "npz_path": str(out_npz),
                    "num_slices": int(x.shape[0]),
                    "num_tumor": int((y == 1).sum()),
                    "num_normal": int((y == 0).sum()),
                }
            )

            if k % 50 == 0:
                print(f"Processed {k}/{len(case_members)} cases...")

    index_df = pd.DataFrame(index_rows).sort_values(["site", "case_id"])
    index_csv = out_dir / "index.csv"
    index_df.to_csv(index_csv, index=False)

    print("Done.")
    print("Saved NPZ dataset to:", out_dir)
    print("Index:", index_csv)
    print("Skipped (no site):", skipped_no_site)
    print("Skipped (missing files/shape):", skipped_missing)
    print("Total cases saved:", len(index_df))
    print("Total slices saved:", int(index_df["num_slices"].sum()))
    return index_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_dataset_samples(
    index_csv: str = config.INDEX_CSV,
    n_show: int = 8,
    mode: str = "random",
    random_state: int = 0,
):
    """Plot one slice per sampled case (tumor slice if available)."""
    assert Path(index_csv).exists(), f"index.csv not found at: {index_csv}"
    df = pd.read_csv(index_csv)
    df = df[df["num_slices"] > 0].copy()

    if mode == "random":
        df_s = df.sample(n=min(n_show, len(df)), random_state=random_state)
    else:
        df_s = df.head(n_show)

    plt.figure(figsize=(16, 8))
    for i, row in enumerate(df_s.itertuples(index=False), start=1):
        npz_path = Path(row.npz_path)
        assert npz_path.exists(), f"Missing npz: {npz_path}"
        data = np.load(npz_path, allow_pickle=True)
        imgs = data["images"]
        labels = data["labels"]

        tumor_idx = np.where(labels == 1)[0]
        if len(tumor_idx) > 0:
            j = int(tumor_idx[0])
            title = f"{row.case_id} | site {row.site:02d} | tumor"
        else:
            j = 0
            title = f"{row.case_id} | site {row.site:02d} | normal"

        ax = plt.subplot(2, (n_show + 1) // 2, i)
        ax.imshow(imgs[j], cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
