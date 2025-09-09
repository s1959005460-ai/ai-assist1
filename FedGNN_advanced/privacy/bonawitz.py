"""
bonawitz.py

Deterministic, parameterized helper routines for pairwise mask derivation
(compatible with Bonawitz-style additive masking protocols).

This implementation focuses on deterministic, auditable mask generation:
- All magic numbers are in constants.py
- RNG derives masks via HMAC-like sha256-based expand (deterministic)
- No secrets are logged; consumers should ensure they don't print raw bytes
"""

from typing import Dict, Iterable, Sequence, Tuple, Any
import hashlib
import hmac
import math
import struct
import itertools
import torch
import numpy as np
from .. import constants
from .. import dp_rng
from .. import logger

# local convenience aliases
SEED_BYTE_LEN = constants.DEFAULTS.get("SEED_BYTE_LEN", 32)
PRG_MASK_BYTES = constants.DEFAULTS.get("PRG_MASK_BYTES", 32)
MASK_SCALE = constants.DEFAULTS.get("MASK_SCALE", 1e-3)


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    Simple HKDF-Expand using HMAC-SHA256 (not full HKDF; sufficient for deterministic expand).
    prk: pseudo-random key (bytes)
    info: context bytes
    length: requested output length in bytes
    """
    out = b""
    t = b""
    i = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), digestmod=hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


def derive_mask_for_param(seed_bytes: bytes, param_path: str, shape: Sequence[int]) -> torch.Tensor:
    """
    Deterministically derive an additive mask for a model parameter.
    - seed_bytes: common seed material (bytes) - treat as secret outside
    - param_path: string to diversify (e.g. 'layer1.weight')
    - shape: desired shape (tuple/list)
    Returns a torch.FloatTensor of the same shape.
    """
    if not isinstance(seed_bytes, (bytes, bytearray)):
        raise TypeError("seed_bytes must be bytes-like")

    info = param_path.encode("utf-8")
    # derive bytes deterministic using HKDF-expand
    n_elems = int(np.prod(shape)) if len(shape) > 0 else 1
    out_bytes_len = max(PRG_MASK_BYTES, n_elems * 4)  # ensure enough bytes for float32 values
    prk = hashlib.sha256(seed_bytes).digest()
    raw = _hkdf_expand(prk, info, out_bytes_len)

    # convert raw bytes to float32 in [-MASK_SCALE, MASK_SCALE]
    # Interpret raw as uint32 chunks
    uint32s = np.frombuffer(raw[: n_elems * 4], dtype=np.uint32)
    # normalize to [0,1)
    normalized = uint32s.astype(np.float64) / float(2**32)
    scaled = (normalized * 2.0 - 1.0) * MASK_SCALE
    arr = scaled.reshape(shape).astype(np.float32)
    return torch.from_numpy(arr)


def derive_pairwise_masks(seed_bytes: bytes, param_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, torch.Tensor]:
    """
    Derive masks for all parameters deterministically.
    param_shapes: mapping param_name -> shape tuple
    Returns dict param_name -> tensor mask
    """
    masks = {}
    for name, shape in param_shapes.items():
        masks[name] = derive_mask_for_param(seed_bytes, name, shape)
    return masks


def example_usage():
    """
    Example (non-secret) usage for testing/demo.
    IMPORTANT: real secret seed_bytes must be kept private.
    """
    seed = b"\x00" * SEED_BYTE_LEN
    shapes = {"w": (2, 3), "b": (3,)}
    masks = derive_pairwise_masks(seed, shapes)
    return masks
