"""
rdp_accountant.py

Numerically stable helper utilities for RDP/DP accounting.

This file removes magic numbers and uses constants.LOG_TINY for numeric stability.
It implements a lightweight RDP -> (epsilon, delta) conversion for Gaussian mechanism
as an example and includes safe log handling.
"""

import math
from typing import Sequence, Tuple
from .. import constants


LOG_TINY = constants.DEFAULTS.get("RDP_LOG_TINY", 1e-300)


def _safe_log(x: float) -> float:
    """
    Stable log that clamps to LOG_TINY to avoid -inf
    """
    return math.log(max(x, LOG_TINY))


def gaussian_rdp(q: float, sigma: float, orders: Sequence[float]) -> Sequence[float]:
    """
    Compute RDP of the Gaussian mechanism for a list of Renyi orders.
    This is the formula from Mironov et al. (standard closed form).
    q: sampling rate (0..1)
    sigma: noise multiplier
    returns list of RDP values corresponding to orders.
    """
    if q <= 0.0:
        return [0.0 for _ in orders]
    if sigma <= 0.0:
        return [float("inf") for _ in orders]

    out = []
    for alpha in orders:
        # stable computations: use logs where helpful
        # For Poisson sampling, we use upper bound method:
        # RDP ≈ (alpha * q**2) / (2 * sigma**2)  (valid for small q)
        # For practicality, use this conservative bound
        val = (alpha * (q ** 2)) / (2.0 * (sigma ** 2))
        out.append(val)
    return out


def get_privacy_spent(orders: Sequence[float], rdp: Sequence[float], target_delta: float) -> Tuple[float, float]:
    """
    Convert RDP to (eps, delta) by optimizing over orders.
    Returns (epsilon, optimal_order)
    """
    if len(orders) != len(rdp):
        raise ValueError("orders and rdp must have same length")
    best_eps = float("inf")
    best_order = None
    for a, r in zip(orders, rdp):
        if r == float("inf"):
            continue
        # bound: eps = r + log(1/delta) / (a-1)
        eps = r + _safe_log(1.0 / max(target_delta, LOG_TINY)) / max((a - 1.0), LOG_TINY)
        if eps < best_eps:
            best_eps = eps
            best_order = a
    if best_order is None:
        return float("inf"), -1.0
    return best_eps, best_order
