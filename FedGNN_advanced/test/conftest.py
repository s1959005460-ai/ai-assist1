import pytest
from FedGNN_advanced import dp_rng

@pytest.fixture(autouse=True)
def set_seed():
    # deterministic seed for tests
    dp_rng.set_seed(12345)
    yield
