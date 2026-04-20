"""MixStyle module and ResNet-18 backbone wrapper.

Reference
---------
Zhou et al. *Domain Generalization with MixStyle*. ICLR 2021.
https://arxiv.org/abs/2104.02008
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torchvision import models


class MixStyle(nn.Module):
    """MixStyle feature-statistics mixing.

    Parameters
    ----------
    p : float
        Probability of applying MixStyle on a given batch.
    alpha : float
        Parameter of the Beta distribution used to sample the mixing weight.
    eps : float
        Numerical stability term when normalising.
    mix : {"random", "crossdomain"}
        Permutation strategy for picking the style donor.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, mix: str = "random"):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self) -> str:
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"

    def set_activation_status(self, status: bool = True) -> None:
        self._activated = status

    def update_mix_method(self, mix: str = "random") -> None:
        self.mix = mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)

        if self.mix == "random":
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            perm = torch.arange(B - 1, -1, -1)
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class ResNet18MixStyle(nn.Module):
    """ResNet-18 backbone with optional MixStyle modules inserted after layers."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        insert_after: tuple[str, ...] = ("layer1", "layer2"),
    ):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)

        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

        self.ms1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if "layer1" in insert_after else None
        self.ms2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if "layer2" in insert_after else None
        self.ms3 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if "layer3" in insert_after else None
        self.ms4 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if "layer4" in insert_after else None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_multiscale: bool = False,
    ):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if self.ms1 is not None:
            x = self.ms1(x)
        l1_out = x

        x = self.backbone.layer2(x)
        if self.ms2 is not None:
            x = self.ms2(x)
        l2_out = x

        x = self.backbone.layer3(x)
        if self.ms3 is not None:
            x = self.ms3(x)
        l3_out = x

        x = self.backbone.layer4(x)
        if self.ms4 is not None:
            x = self.ms4(x)
        l4_out = x

        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.backbone.fc(feats)

        if return_multiscale:
            inter = {
                "layer1": l1_out,
                "layer2": l2_out,
                "layer3": l3_out,
                "layer4": l4_out,
                "feats": feats,
            }
            return logits, feats, inter
        if return_features:
            return logits, feats
        return logits
