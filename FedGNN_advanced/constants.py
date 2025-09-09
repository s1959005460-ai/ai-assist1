"""
constants.py

Centralized constants for FedGNN_advanced. This version includes fields
used by the Bonawitz / Shamir implementation.
"""

from typing import Dict, Any
import numpy as np

DEFAULTS: Dict[str, Any] = {
    # RNG / seed handling
    "DEFAULT_SEED": 0,

    # Masking / Bonawitz related defaults
    "MASK_SCALE": 1e-3,
    "PRG_MASK_BYTES": 32,
    "SEED_BYTE_LEN": 32,

    # TopoReg defaults
    "TOPO_MARGIN": 1.0,
    "TOPO_K": 5,
    "TOPO_REDUCTION": "mean",

    # Numeric stability
    "LOG_TINY": np.finfo(float).tiny,
    "EPS": 1e-12,

    # RDP / DP accountant
    "RDP_LOG_TINY": 1e-300,
    "DP_CLIP_NORM": 1.0,

    # Aggregator
    "AGG_MIN_TOTAL_WEIGHT": 1e-12,

    # Networking / defaults
    "DEFAULT_PORT": 10000,
    "DEFAULT_HOST": "127.0.0.1",

    # Large constants
    "LARGE_NEG": -1e30,
    "LARGE_POS": 1e30,

    # SHAMIR / finite field for secret sharing (large prime)
    # Use a 127-bit Mersenne-like prime: 2**127 - 1
    # Python supports big integers; this prime is large enough for secrets up to 32 bytes.
    "SHAMIR_PRIME": (1 << 127) - 1,

    # Default threshold policy (fraction of participants needed to reconstruct)
    # Typical Bonawitz uses threshold = number_of_participants - tolerated_dropouts
    # Here we expose a default fraction; concrete threshold computed per-run.
    "SHAMIR_THRESHOLD_FRACTION": 0.5,  # meaning need at least ceil(n * fraction)
}

# convenience accessors
LOG_TINY = DEFAULTS["LOG_TINY"]
EPS = DEFAULTS["EPS"]
AGG_MIN_TOTAL_WEIGHT = DEFAULTS["AGG_MIN_TOTAL_WEIGHT"]


def update_from_dict(d):
    DEFAULTS.update(d)
    # update convenience names
    globals()["LOG_TINY"] = DEFAULTS["LOG_TINY"]
    globals()["EPS"] = DEFAULTS["EPS"]
    globals()["AGG_MIN_TOTAL_WEIGHT"] = DEFAULTS["AGG_MIN_TOTAL_WEIGHT"]
