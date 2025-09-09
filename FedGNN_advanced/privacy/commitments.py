# privacy/commitments.py
"""
Pedersen commitment + Schnorr-style non-interactive ZK proof (Fiat-Shamir) for knowledge of opening.

注意:
 - 这是一个教学/原型实现。生产环境必须使用经过审核的参数与实现。
 - PRIME_FIELD, G, H 在此模块内部定义，保证模块独立（避免循环 import）。
"""

import os
import hashlib
from typing import Tuple, Dict, Any

# large prime field (use a large safe prime; here we use 2^521-1 as prototype)
PRIME_FIELD = 2**521 - 1

# simple small generator g (for demonstration). In production choose standard group params.
G = 2

def derive_h_from_seed(seed=b'commitment-h'):
    s = hashlib.sha256(seed).digest()
    h = int.from_bytes(s, 'big') % PRIME_FIELD
    if h == 0:
        h = 3
    return h

H = derive_h_from_seed(b'pedersen-h-generator')

def pedersen_commit(m: int, r: int, g: int = G, h: int = H, p: int = PRIME_FIELD) -> int:
    """
    c = g^m * h^r mod p
    """
    gm = pow(g, m % p, p)
    hr = pow(h, r % p, p)
    return (gm * hr) % p

def create_schnorr_proof(m: int, r: int, g: int = G, h: int = H, p: int = PRIME_FIELD) -> Dict[str, Any]:
    """
    Create Schnorr-style NIZK (Fiat-Shamir) proving knowledge of m,r s.t. c = g^m h^r.
    Returns proof dict with fields (c, t, s1, s2).
    """
    # random k1,k2
    k1 = int.from_bytes(os.urandom(32), 'big') % p
    k2 = int.from_bytes(os.urandom(32), 'big') % p
    t = (pow(g, k1, p) * pow(h, k2, p)) % p
    c = pedersen_commit(m, r, g, h, p)
    e_bytes = hashlib.sha256((str(c) + '|' + str(t)).encode('utf-8')).digest()
    e = int.from_bytes(e_bytes, 'big') % p
    s1 = (k1 - (e * m)) % p
    s2 = (k2 - (e * r)) % p
    proof = {'c': int(c), 't': int(t), 's1': int(s1), 's2': int(s2)}
    return proof

def verify_schnorr_proof(proof: Dict[str, Any], g: int = G, h: int = H, p: int = PRIME_FIELD) -> bool:
    """
    Verify Schnorr-style proof. Check: g^{s1} h^{s2} * c^{e} == t  (mod p)
    where e = H(c || t)
    """
    c = proof['c']
    t = proof['t']
    s1 = proof['s1']
    s2 = proof['s2']
    e_bytes = hashlib.sha256((str(c) + '|' + str(t)).encode('utf-8')).digest()
    e = int.from_bytes(e_bytes, 'big') % p
    left = (pow(g, s1, p) * pow(h, s2, p) * pow(c, e, p)) % p
    return left == t

# Helpers to convert between ints and bytes (for seeds)
def int_from_bytes(b: bytes, p: int = PRIME_FIELD) -> int:
    return int.from_bytes(b, 'big') % p

def bytes_from_int(i: int, length: int = None) -> bytes:
    blen = (i.bit_length() + 7) // 8 or 1
    if length is None:
        return i.to_bytes(blen, 'big')
    return i.to_bytes(length, 'big')
