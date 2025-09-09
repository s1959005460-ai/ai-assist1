# privacy/dp.py
"""
差分隐私工具：L2 裁剪 + 高斯噪声添加（用于服务器在聚合后施加噪声）
注意：这是实验/教学实现，若要严格 (epsilon, delta) 计量/会计，需要使用 Opacus / RDP accountant。
"""
import torch
import math
from typing import Dict

def clip_state_dict(state: Dict[str, torch.Tensor], max_norm: float):
    total = 0.0
    for v in state.values():
        total += (v.view(-1).float().pow(2).sum().item())
    norm = math.sqrt(total)
    if norm <= max_norm:
        return state, norm
    scale = max_norm / (norm + 1e-12)
    clipped = {k: (v * scale).clone() for k,v in state.items()}
    return clipped, norm

def add_gaussian_noise_to_state(state: Dict[str, torch.Tensor], noise_std: float):
    noised = {}
    for k, v in state.items():
        noise = torch.normal(mean=0.0, std=noise_std, size=v.shape, device=v.device, dtype=v.dtype)
        noised[k] = v + noise
    return noised

def apply_dp_to_aggregated_delta(aggregated_delta: Dict[str, torch.Tensor], max_norm: float, noise_multiplier: float, sample_factor: float=1.0):
    """
    aggregated_delta: dict tensors on device
    max_norm: clipping norm (L2)
    noise_multiplier: sigma (relative to max_norm)
    sample_factor: optional scaling factor due to sampling/prob
    Returns noised aggregated_delta
    """
    # clip
    clipped, norm = clip_state_dict(aggregated_delta, max_norm)
    # compute std: noise_multiplier * max_norm * sqrt(1/sample_factor) (approx)
    std = noise_multiplier * max_norm * math.sqrt(1.0 / max(1e-12, sample_factor))
    noised = add_gaussian_noise_to_state(clipped, std)
    return noised
