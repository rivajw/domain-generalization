"""Shared runtime state: device, AMP flag, and gradient scaler.

The module defers importing ``torch`` until one of the runtime attributes is
actually accessed. That keeps notebook setup cells lightweight and avoids
dragging PyTorch into the import path before the environment is settled.
"""

from __future__ import annotations

from typing import Any

_torch = None
_DEVICE = None
_USE_AMP = None
_scaler = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch as imported_torch

        _torch = imported_torch
    return _torch


def _ensure_runtime() -> None:
    global _DEVICE, _USE_AMP
    if _DEVICE is None or _USE_AMP is None:
        torch = _get_torch()
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _USE_AMP = _DEVICE.type == "cuda"


def reset_scaler():
    """Create a fresh GradScaler (call before each experiment run)."""
    global _scaler
    _ensure_runtime()
    torch = _get_torch()
    _scaler = torch.amp.GradScaler("cuda", enabled=_USE_AMP)
    return _scaler


def __getattr__(name: str) -> Any:
    if name in {"DEVICE", "USE_AMP"}:
        _ensure_runtime()
        return _DEVICE if name == "DEVICE" else _USE_AMP
    if name == "scaler":
        if _scaler is None:
            reset_scaler()
        return _scaler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
