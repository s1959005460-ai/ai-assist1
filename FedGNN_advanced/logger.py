"""
logger.py

Centralized logging helpers with sanitization to avoid accidentally logging secrets.

This file is safe to place at repo root (FedGNN_advanced/logger.py).
"""

import logging
from typing import Any
from . import constants

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("FedGNN")

_SENSITIVE_KEYS = {"secret", "private_key", "privkey", "seed", "sk", "sk_bytes", "password", "token", "salt"}


def _sanitize(obj: Any) -> Any:
    """
    Recursively sanitize common containers to avoid logging secrets.
    """
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if str(k).lower() in _SENSITIVE_KEYS:
                    out[k] = "<REDACTED>"
                else:
                    out[k] = _sanitize(v)
            return out
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_sanitize(x) for x in obj)
        else:
            return obj
    except Exception:
        return "<UNSANITIZABLE>"


def secure_log(level: str, msg: str, **kwargs):
    """
    Log while sanitizing kwargs.
    Example: secure_log('info','starting', extra={'secret': b'xxx'})
    """
    lg = getattr(logger, level.lower(), logger.info)
    sanitized = {k: _sanitize(v) for k, v in kwargs.items()}
    lg(msg + " | " + str(sanitized))


def secure_zero(buffer) -> bool:
    """
    Best-effort overwrite for mutable buffers (bytearray / memoryview).
    Returns True on success, False otherwise.
    """
    try:
        if isinstance(buffer, bytearray):
            for i in range(len(buffer)):
                buffer[i] = 0
            return True
        if isinstance(buffer, memoryview) and not buffer.readonly:
            buffer[:] = b"\x00" * len(buffer)
            return True
        return False
    except Exception:
        return False
