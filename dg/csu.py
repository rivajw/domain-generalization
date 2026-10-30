"""Correlated Style Uncertainty (CSU), adapted for the project's ResNet18 pipeline.

Reference: Zhang et al., Domain Generalization with Correlated Style
Uncertainty, WACV 2024, Eqs. (1)-(4), (11)-(16).
https://openaccess.thecvf.com/content/WACV2024/html/Zhang_Domain_Generalization_With_Correlated_Style_Uncertainty_WACV_2024_paper.html

This implements the published covariance/eigendecomposition formulation
independently, not copied from the authors' repository. Style statistics and
covariance are detached as in MixStyle; features retain gradients.
"""
from __future__ import annotations

import random

import torch
from torch import nn


class CSU(nn.Module):
    """Correlated feature-statistic uncertainty on [B,C,H,W] feature maps.

    p: batch-level activation probability.
    alpha: Beta(alpha, alpha) amplitude, sampled independently per image.
    eps: floor for instance standard deviations.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("CSU p must be between 0 and 1")
        if alpha <= 0:
            raise ValueError("CSU alpha must be > 0")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(alpha, alpha)
        self._activated = True

    def __repr__(self):
        return f"CSU(p={self.p}, alpha={self.alpha}, eps={self.eps})"

    def set_activation_status(self, status: bool = True):
        self._activated = bool(status)

    @staticmethod
    @torch.no_grad()
    def _correlated_noise(stats: torch.Tensor) -> torch.Tensor:
        """Sample N(0, Cov(stats)) using symmetric square root Q sqrt(L) Q.T.

        stats: float32 tensor [B,C]. Uses population covariance / B,
        following Eq. (3)-(4) of the CSU paper. Batch size must be >=2.
        """
        b = stats.shape[0]
        centered = stats - stats.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / b
        cov = (cov + cov.T) * 0.5  # numerical symmetry
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(0)
        # (Q diag(sqrt(eigenvalues)) Q^T) acting on independent noise.
        noise = torch.randn_like(stats)
        projection = noise @ eigvecs
        projection *= eigvals.sqrt().unsqueeze(0)
        return projection @ eigvecs.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated or random.random() > self.p:
            return x
        if x.ndim != 4:
            raise ValueError("CSU expects a 4-D [batch, channels, height, width] tensor")
        b = x.shape[0]
        if b < 2:  # covariance is not meaningful for a singleton batch
            return x
        # Eigh is unsupported for float16/bfloat16; float32 improves stability
        # under the experiment's CUDA autocast (AMP) training loop.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            with torch.no_grad():
                mu = x_float.mean(dim=(2, 3), keepdim=True)
                sigma = (x_float.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt()
                delta_mu = self._correlated_noise(mu.flatten(1))
                delta_sigma = self._correlated_noise(sigma.flatten(1))
                amplitude = self.beta.sample((b, 1)).to(device=x.device, dtype=torch.float32)
                new_mu = mu + (amplitude * delta_mu).view(b, -1, 1, 1)
                new_sigma = sigma + (amplitude * delta_sigma).view(b, -1, 1, 1)
            output = ((x_float - mu) / sigma) * new_sigma + new_mu
        return output.to(dtype=x.dtype)
