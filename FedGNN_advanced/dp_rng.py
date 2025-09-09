"""
dp_rng.py

Central RNG utilities for reproducible DP/noise operations and to remove magic seeds.

Usage:
    import dp_rng
    dp_rng.set_seed(42)
    gen = dp_rng.get_torch_generator()
    rng = dp_rng.get_numpy_rng()
"""

import random
import torch
import numpy as np
from typing import Optional
from . import constants

_torch_gen: Optional[torch.Generator] = None
_numpy_rng: Optional[np.random.Generator] = None
_python_random_state_set: bool = False
_current_seed: Optional[int] = None


def set_seed(seed: Optional[int]):
    """
    Set global seed for python.random, numpy and torch.
    Use this function once at experiment bootstrap.
    """
    global _torch_gen, _numpy_rng, _current_seed, _python_random_state_set
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    _current_seed = int(seed)

    # python random
    random.seed(_current_seed)
    _python_random_state_set = True

    # numpy
    _numpy_rng = np.random.default_rng(_current_seed)

    # torch
    _torch_gen = torch.Generator()
    _torch_gen.manual_seed(_current_seed)


def get_torch_generator() -> torch.Generator:
    """
    Return a torch.Generator seeded via set_seed. If not set, set default seed.
    """
    global _torch_gen
    if _torch_gen is None:
        set_seed(constants.DEFAULTS.get("DEFAULT_SEED", 0))
    return _torch_gen


def get_numpy_rng() -> np.random.Generator:
    global _numpy_rng
    if _numpy_rng is None:
        set_seed(constants.DEFAULTS.get("DEFAULT_SEED", 0))
    return _numpy_rng


def current_seed() -> Optional[int]:
    return _current_seed
