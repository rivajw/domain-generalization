"""Configuration constants for BraTS domain-generalization experiments."""

import os
from pathlib import Path


def _load_synapse_token() -> str:
    """Read SYNAPSE_PAT from .secrets.env (gitignored) or environment."""
    # Check environment variable first
    val = os.environ.get("SYNAPSE_PAT", "")
    if val:
        return val
    # Fall back to .secrets.env next to the project root
    secrets_path = Path(__file__).resolve().parent.parent / ".secrets.env"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("SYNAPSE_PAT="):
                return line.split("=", 1)[1].strip()
    return ""


# -----------------------
# Paths
# -----------------------
TOKEN = _load_synapse_token()
ZIP_PATH = "./ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
MAPPING_XLSX = "./BraTS2023_2017_GLI_Mapping.xlsx"
OUT_ROOT = "./brats23_npz_dg"
INDEX_CSV = "./brats23_npz_dg/index.csv"

# -----------------------
# Modality / preprocessing
# -----------------------
# BraTS modalities available: t1c, t1n, t2w, t2f
# t2f ~ FLAIR is a good default for tumor visibility
MODALITY = "t2f"
AXIS = 2  # axial
MIN_BRAIN_PIXELS = 500
TUMOR_PIXEL_THRESHOLD = 20

# -----------------------
# Dataset-size control
# -----------------------
VALIDATION_FRACTION = 0.15
IN_DOMAIN_TEST_FRACTION = 0.15
MAX_TUMOR_SLICES = 40
MAX_NORMAL_SLICES = 40
MAX_CASES_PER_SITE = 64   # cap seen-domain cases per site before split to prevent bias toward large sites
MIN_CASES_FOR_LOSO = 30   # only rank held-out sites with enough cases for a leave-one-site-out eval
OUT_SIZE = (224, 224)     # NPZ slices stored as 224x224 uint8

# -----------------------
# Training
# -----------------------
EXP_SEED = 18662
EARLY_STOP_PATIENCE = 3
