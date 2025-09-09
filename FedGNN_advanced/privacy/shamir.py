"""
shamir.py

Simple Shamir secret sharing over a large prime field.

Functions:
- split_secret_bytes(secret_bytes, n, t, prime) -> List[(idx:int, share:int)]
- reconstruct_secret_bytes(shares, secret_len_bytes, prime) -> secret_bytes

Design notes:
- Secrets are treated as big integers via int.from_bytes(..., 'big').
- Prime must be > secret_int. We use constants.DEFAULTS['SHAMIR_PRIME'] by default.
- This implementation is educational / prototype quality and should be reviewed
  for production cryptographic deployment.
"""

from typing import List, Tuple
import random
from .. import constants
from .. import dp_rng

PRIME = constants.DEFAULTS.get("SHAMIR_PRIME", (1 << 127) - 1)


def _eval_polynomial(coeffs: List[int], x: int, prime: int) -> int:
    """Evaluate polynomial with integer coefficients at point x modulo prime"""
    res = 0
    for a in reversed(coeffs):
        res = (res * x + a) % prime
    return res


def split_secret_bytes(secret: bytes, n: int, t: int, prime: int = None) -> List[Tuple[int, int]]:
    """
    Split secret bytes into n shares with threshold t.
    Returns list of (i, share_int) with i in [1..n].
    """
    if prime is None:
        prime = PRIME
    secret_int = int.from_bytes(secret, byteorder="big")
    if secret_int >= prime:
        raise ValueError("Secret must be smaller than prime field")
    if not (1 <= t <= n):
        raise ValueError("Threshold t must satisfy 1 <= t <= n")

    # generate random polynomial coefficients: degree t-1
    rng = dp_rng.get_numpy_rng()
    # coefficients: a_0 = secret_int, a_1..a_{t-1} random in [0, prime-1]
    coeffs = [secret_int] + [int(rng.integers(0, prime)) for _ in range(t - 1)]
    shares = []
    for i in range(1, n + 1):
        x = i
        y = _eval_polynomial(coeffs, x, prime)
        shares.append((i, y))
    return shares


def _lagrange_interpolate(x: int, xs: List[int], ys: List[int], prime: int) -> int:
    """
    Compute Lagrange interpolation at point x given points xs, ys, modulo prime.
    Returns integer in [0, prime).
    """
    assert len(xs) == len(ys)
    total = 0
    k = len(xs)
    for j in range(k):
        numerator = 1
        denominator = 1
        xj = xs[j]
        for m in range(k):
            if m == j:
                continue
            xm = xs[m]
            numerator = (numerator * (x - xm)) % prime
            denominator = (denominator * (xj - xm)) % prime
        # compute term = yj * numerator * inv(denominator)
        inv_den = pow(denominator, -1, prime)
        term = ys[j] * numerator * inv_den
        total = (total + term) % prime
    return total


def reconstruct_secret_bytes(shares: List[Tuple[int, int]], secret_len: int, prime: int = None) -> bytes:
    """
    Reconstruct secret bytes from shares: list of (i, share_int). secret_len is the original byte length.
    """
    if prime is None:
        prime = PRIME
    if len(shares) == 0:
        raise ValueError("Need at least one share")
    xs = [s[0] for s in shares]
    ys = [s[1] for s in shares]
    secret_int = _lagrange_interpolate(0, xs, ys, prime)
    # convert to bytes with the original length
    secret_bytes = secret_int.to_bytes(secret_len, byteorder="big")
    return secret_bytes
