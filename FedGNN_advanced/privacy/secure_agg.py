"""
secure_agg.py

Utilities for secure aggregation that avoid global np.random.seed and magic seeds.
Uses dp_rng for RNG.
"""

import numpy as np
from .. import dp_rng
from .. import constants

def random_mask(shape, scale=None):
    rng = dp_rng.get_numpy_rng()
    if scale is None:
        scale = constants.DEFAULTS.get("MASK_SCALE", 1e-3)
    return rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float32)
