#!/usr/bin/env python3
"""Numerical checks against published CSU equations (not author's code parity)."""
import numpy as np
import torch
from dg.csu import CSU


def main():
    rng=np.random.default_rng(123)
    means=rng.normal(size=(40, 8)).astype('float64')
    means[:, 2] = 0.8 * means[:, 0] + 0.6 * means[:, 2]  # induce covariance
    stats=torch.tensor(means, dtype=torch.float64)
    centered=stats-stats.mean(0)
    cov=(centered.T@centered)/len(stats)
    expected=np.cov(means,rowvar=False,bias=True)
    assert np.allclose(cov.numpy(),expected,atol=1e-12)
    lam,Q=torch.linalg.eigh(cov)
    P=(Q*lam.clamp_min(0).sqrt()[None,:])@Q.T
    assert torch.allclose(P@P.T,cov,atol=1e-10)
    torch.manual_seed(42)
    eps=torch.randn(30000,8,dtype=torch.float64)
    correlated=eps@P
    empirical=torch.cov(correlated.T,correction=0)
    relative=(empirical-cov).norm()/cov.norm()
    assert relative<0.05, relative
    module=CSU(p=1,alpha=0.1)
    x=torch.randn(8,8,6,6,requires_grad=True)
    out=module(x)
    assert out.shape==x.shape and torch.isfinite(out).all()
    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    module.eval()
    assert torch.equal(module(x.detach()),x.detach())
    print('PASS covariance normalization (1/B) vs NumPy')
    print('PASS symmetric covariance square root P@P.T=Cov')
    print('PASS sampled correlation, relative error:',round(relative.item(),4))
    print('PASS forward, backward, evaluation identity')
    print('NOTE: Tests match WACV 2024 mathematical equations, NOT exact official code execution.')

if __name__=='__main__':
    main()
