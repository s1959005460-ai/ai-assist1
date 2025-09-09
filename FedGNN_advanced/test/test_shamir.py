import pytest
from FedGNN_advanced.privacy.shamir import split_secret_bytes, reconstruct_secret_bytes
from FedGNN_advanced import constants


def test_shamir_split_and_reconstruct():
    # secret length equals SEED_BYTE_LEN
    secret_len = constants.DEFAULTS.get("SEED_BYTE_LEN", 32)
    secret = b"\x01\x23" * (secret_len // 2)
    n = 6
    t = 3

    shares = split_secret_bytes(secret, n=n, t=t)
    assert len(shares) == n

    # pick any t shares
    chosen = shares[1:1 + t]
    pairs = [(int(idx), int(val)) for idx, val in chosen]

    rec = reconstruct_secret_bytes(pairs, secret_len)
    assert rec == secret


def test_shamir_secret_too_large_raises():
    prime = constants.DEFAULTS.get("SHAMIR_PRIME")
    # create a secret that equals prime (invalid)
    secret_int = prime
    secret_len = constants.DEFAULTS.get("SEED_BYTE_LEN", 32)
    secret = secret_int.to_bytes(secret_len, byteorder="big")
    with pytest.raises(ValueError):
        split_secret_bytes(secret, n=5, t=3)
