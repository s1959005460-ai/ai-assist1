# tests/test_dp.py
import torch
from privacy.dp import clip_and_noise_delta

def test_dp_clipping_and_noise():
    d = {'w': torch.randn(10,10)}
    noisy, norm = clip_and_noise_delta(d, max_norm=0.5, noise_multiplier=0.1, seed=42)
    assert isinstance(noisy, dict)
    assert 'w' in noisy
