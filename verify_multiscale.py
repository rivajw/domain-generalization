#!/usr/bin/env python3
"""Quick standalone tests; no BraTS data or pretrained downloads required."""
import torch
import random
from dg.models import ResNet18MixStyle
from dg.consistency import set_all_mixstyle
from dg.multiscale import STAGE_NAMES, capture_multiscale_features, multiscale_cosine_loss


def main():
    torch.manual_seed(25)
    random.seed(25)
    model = ResNet18MixStyle(pretrained=False, num_classes=2,
                             insert_after=('layer2',), mixstyle_p=1.0, mixstyle_alpha=0.5)
    model.train()
    x = torch.randn(4, 3, 64, 64)
    with capture_multiscale_features(model) as features:
        with torch.no_grad():
            set_all_mixstyle(model, False)
            logits_clean = model(x)
            clean = {k: features[k].detach().clone() for k in STAGE_NAMES}
            assert list(clean) == list(STAGE_NAMES)
            features.clear()
            # OFF -> OFF must give ~0 consistency loss
            model(x)
            zero, _ = multiscale_cosine_loss(features, clean)
            assert abs(zero.item()) < 2e-5, zero.item()
        features.clear()
        set_all_mixstyle(model, True)
        logits = model(x)
        loss, parts = multiscale_cosine_loss(features, clean)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert loss.item() >= -2e-5, loss.item()
        # Mixed path must generally change layer2 and downstream embeddings.
        assert parts['layer2_postmix'].item() > 1e-6
        assert logits.shape == (4,2) == logits_clean.shape
        loss.backward()
        assert model.backbone.layer2[0].conv1.weight.grad is not None
        print('cosine stages:', {k: round(v.item(), 6) for k,v in parts.items()})
        print('Loss:', loss.item(), 'grad:', model.backbone.layer2[0].conv1.weight.grad.abs().mean().item())
    assert len(model.backbone.layer3._forward_hooks) == 0
    model.eval()
    with torch.no_grad():
        eval_logits = model(x)
    assert torch.isfinite(eval_logits).all()
    print('PASS: shapes, clean identity, mixed perturbation, gradients, hook cleanup, eval.')

if __name__ == '__main__':
    main()
