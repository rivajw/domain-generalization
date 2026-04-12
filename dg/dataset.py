"""PyTorch Dataset that loads per-case NPZ files and exposes slice-level samples."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSNPZSliceDataset(Dataset):
    """Loads per-case .npz files and exposes slice-level samples.

    Caches all arrays in memory to reduce disk I/O.

    Returns
    -------
    x : FloatTensor [3, 224, 224] in [0, 1]
    y : LongTensor scalar (0/1)
    site : int (domain id)
    case_id : str
    """

    def __init__(self, npz_paths):
        self.npz_paths = list(npz_paths)
        self._files = []
        self._index = []
        for fi, p in enumerate(self.npz_paths):
            with np.load(p) as d:
                images = d["images"].astype(np.uint8, copy=False)
                labels = d["labels"].astype(np.int64, copy=False)
                site = int(d["site"])
                case_id = str(d["case_id"])
            self._files.append(
                {
                    "images": images,
                    "labels": labels,
                    "site": site,
                    "case_id": case_id,
                }
            )
            self._index.extend([(fi, si) for si in range(images.shape[0])])

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, si = self._index[idx]
        f = self._files[fi]
        img = f["images"][si]  # [224,224] uint8
        y = int(f["labels"][si])

        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        x = x.repeat(3, 1, 1)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, f["site"], f["case_id"]
