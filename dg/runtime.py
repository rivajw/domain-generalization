"""Shared runtime state: device, AMP flag, and gradient scaler.

These are module-level so training/eval helpers can read the same values
without passing them as arguments. Call :func:`reset_scaler` at the start
of each experiment to get a fresh GradScaler.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


def reset_scaler():
    """Create a fresh GradScaler (call before each experiment run)."""
    global scaler
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
