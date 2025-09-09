# utils/serialization.py
"""
Efficient serialization utilities for model state_dict:
 - serialize_state_dict: pack whole state_dict into bytes via torch.save(buffer)
 - chunk_bytes: generator to yield chunks for streaming
 - deserialize_state_bytes: recover state_dict from bytes
"""
import io
import torch
from typing import Dict, Generator

def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    # use torch.save once (entire dict) - more efficient than saving per-tensor over RPC
    torch.save(state_dict, buf)
    return buf.getvalue()

def deserialize_state_bytes(b: bytes):
    buf = io.BytesIO(b)
    s = torch.load(buf, map_location='cpu')
    return s

def chunk_bytes(b: bytes, chunk_size: int = 64 * 1024) -> Generator[bytes, None, None]:
    for i in range(0, len(b), chunk_size):
        yield b[i:i+chunk_size]
