"""
mask_manager.py

Utilities to deterministically derive pairwise seeds and to construct masks for model params.

- derive_pairwise_seed(client_a, client_b, master_seed)
    returns bytes of length SEED_BYTE_LEN agreed by both parties.

- compute_mask_from_seed(seed_bytes, param_shapes) -> dict param_name -> numpy float32 array
    (caller can convert to torch if needed)
"""

import hashlib
import hmac
from typing import Dict, Sequence, Tuple, Any
import numpy as np
from .. import constants
from .. import dp_rng

SEED_BYTE_LEN = constants.DEFAULTS.get("SEED_BYTE_LEN", 32)
PRG_MASK_BYTES = constants.DEFAULTS.get("PRG_MASK_BYTES", 32)
MASK_SCALE = constants.DEFAULTS.get("MASK_SCALE", 1e-3)


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """
    Basic HKDF-expand (HMAC-SHA256) to deterministically expand prk + info -> length bytes.
    """
    out = b""
    t = b""
    i = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), digestmod=hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


def derive_pairwise_seed(client_a: str, client_b: str, master_seed: bytes) -> bytes:
    """
    Deterministically derive a pairwise seed for (a,b). Order-invariant: uses sorted ids.
    master_seed: shared secret seed material (bytes) known to the pair participants (e.g., generated per-client).
    """
    a, b = sorted([str(client_a), str(client_b)])
    info = (a + "|" + b).encode("utf-8")
    prk = hashlib.sha256(master_seed).digest()
    out = _hkdf_expand(prk, info, SEED_BYTE_LEN)
    return out


def compute_mask_from_seed(seed_bytes: bytes, param_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
    """
    Given seed_bytes and desired param shapes, produce deterministic masks as numpy float32 arrays.
    Masks values are in [-MASK_SCALE, MASK_SCALE].
    """
    masks = {}
    prk = hashlib.sha256(seed_bytes).digest()
    for name, shape in param_shapes.items():
        n_elems = int(np.prod(shape)) if len(shape) > 0 else 1
        out_len = max(PRG_MASK_BYTES, n_elems * 4)
        raw = _hkdf_expand(prk, name.encode("utf-8"), out_len)
        uint32s = np.frombuffer(raw[: n_elems * 4], dtype=np.uint32)
        normalized = uint32s.astype(np.float64) / float(2**32)
        scaled = (normalized * 2.0 - 1.0) * MASK_SCALE
        arr = scaled.reshape(shape).astype(np.float32)
        masks[name] = arr
    return masks
