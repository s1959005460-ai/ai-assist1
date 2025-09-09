"""
compression.py

Utilities to serialize tensors (or numpy arrays) with a naive sparse encoding,
and return compression metadata for monitoring.
"""

import io
import pickle
import numpy as np
from typing import Any, Dict, Tuple

def serialize_sparse(tensor: Any) -> Tuple[bytes, Dict[str, Any]]:
    """
    Naive sparse serialization: extract non-zero indices and values and pickle them.
    Returns (bytes_payload, metadata)
    metadata includes uncompressed_bytes, compressed_bytes, ratio
    """
    arr = np.array(tensor)
    uncompressed = arr.nbytes
    # find non-zero entries
    nz = np.nonzero(arr)
    if arr.ndim == 0:
        payload = {"shape": (), "idx": np.array([], dtype=np.int32), "vals": np.array([arr.item()], dtype=np.float32)}
    else:
        idx = np.vstack(nz).T.astype(np.int32)
        vals = arr[nz].astype(np.float32)
        payload = {"shape": arr.shape, "idx": idx, "vals": vals}
    b = pickle.dumps(payload, protocol=4)
    compressed = len(b)
    ratio = compressed / max(1, uncompressed)
    meta = {"uncompressed_bytes": uncompressed, "compressed_bytes": compressed, "ratio": ratio, "nnz": payload.get("idx").shape[0] if "idx" in payload else 1}
    return b, meta


def deserialize_sparse(b: bytes) -> Any:
    payload = pickle.loads(b)
    shape = payload["shape"]
    idx = payload["idx"]
    vals = payload["vals"]
    arr = np.zeros(shape, dtype=np.float32)
    if idx.size > 0:
        arr[tuple(idx.T)] = vals
    return arr
