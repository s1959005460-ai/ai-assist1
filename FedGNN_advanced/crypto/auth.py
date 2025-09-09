# crypto/auth.py
"""
Authentication utilities: Ed25519 signing/verification, signed envelopes and anti-replay.

依赖:
    pip install cryptography
"""

import time
import json
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
except Exception:
    Ed25519PrivateKey = None

def require_crypto():
    if Ed25519PrivateKey is None:
        raise RuntimeError("cryptography not installed. Install with `pip install cryptography`")

# --- Key utilities ---

def generate_ed25519_keypair() -> Tuple[bytes, bytes]:
    """
    Returns (priv_bytes, pub_bytes) raw bytes for Ed25519.
    """
    require_crypto()
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub = priv.public_key()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    return priv_bytes, pub_bytes

def load_private_key(priv_bytes: bytes):
    require_crypto()
    return Ed25519PrivateKey.from_private_bytes(priv_bytes)

def load_public_key(pub_bytes: bytes):
    require_crypto()
    return Ed25519PublicKey.from_public_bytes(pub_bytes)

def sign_message(priv_bytes: bytes, message: bytes) -> bytes:
    """
    Sign raw bytes; returns signature bytes.
    """
    require_crypto()
    priv = load_private_key(priv_bytes)
    sig = priv.sign(message)
    return sig

def verify_signature(pub_bytes: bytes, message: bytes, signature: bytes) -> bool:
    try:
        pub = load_public_key(pub_bytes)
        pub.verify(signature, message)
        return True
    except Exception:
        return False

# --- Structured message helpers (include seq & timestamp to fight replay) ---
def pack_signed_payload(payload: Dict[str, Any], priv_bytes: bytes, seq: int = None, timestamp: float = None) -> Dict[str, Any]:
    """
    Add seq and timestamp, then sign the canonical JSON bytes.
    Returns envelope containing: payload, seq, ts, signature (hex), pub (hex)
    """
    require_crypto()
    p = dict(payload)
    if seq is None:
        seq = int(time.time() * 1000)  # ms timestamp default
    if timestamp is None:
        timestamp = time.time()
    envelope = {
        'payload': p,
        'seq': int(seq),
        'ts': float(timestamp)
    }
    # canonical JSON bytes
    msg = json.dumps(envelope, sort_keys=True, separators=(',', ':')).encode('utf-8')
    sig = sign_message(priv_bytes, msg)
    # include pubkey for verification (in production use certificates / registration)
    priv = load_private_key(priv_bytes)
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    envelope['signature'] = sig.hex()
    envelope['pub'] = pub_bytes.hex()
    return envelope

def unpack_and_verify_envelope(envelope: Dict[str, Any], expected_pub_hex: str = None, allow_time_skew: float = 300.0) -> (bool, Dict[str, Any]):
    """
    Verify signature and timestamp. Returns (ok, payload).
    expected_pub_hex: if provided, envelope['pub'] must match (avoid spoofing).
    """
    require_crypto()
    sig_hex = envelope.get('signature')
    pub_hex = envelope.get('pub')
    seq = envelope.get('seq')
    ts = envelope.get('ts')
    if sig_hex is None or pub_hex is None or seq is None or ts is None:
        return False, {}
    if expected_pub_hex and pub_hex != expected_pub_hex:
        return False, {}
    # reconstruct msg
    msg = {'payload': envelope['payload'], 'seq': int(seq), 'ts': float(ts)}
    msg_bytes = json.dumps(msg, sort_keys=True, separators=(',', ':')).encode('utf-8')
    sig = bytes.fromhex(sig_hex)
    pub_bytes = bytes.fromhex(pub_hex)
    ok = verify_signature(pub_bytes, msg_bytes, sig)
    if not ok:
        return False, {}
    # time skew check
    now = time.time()
    if abs(now - float(ts)) > float(allow_time_skew):
        logger.warning("Timestamp skew too large: now=%s ts=%s", now, ts)
        return False, {}
    return True, envelope['payload']

# --- Simple anti-replay store (server-side) ---
class AntiReplayStore:
    """
    Very simple anti-replay: track highest seq seen per pub (hex).
    Accept only seq > last_seq.
    """
    def __init__(self):
        self._last_seq = {}

    def accept(self, pub_hex: str, seq: int) -> bool:
        last = self._last_seq.get(pub_hex)
        if last is None or int(seq) > int(last):
            self._last_seq[pub_hex] = int(seq)
            return True
        return False

    def get_last_seq(self, pub_hex: str) -> int:
        return self._last_seq.get(pub_hex, 0)
