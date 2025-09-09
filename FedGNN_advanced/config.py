"""
config.py

Pydantic-based configuration models for experiments.
Provides typed configs and parsing from yaml/json.
"""

from pydantic import BaseModel, Field
from typing import Optional

class TrainConfig(BaseModel):
    rounds: int = Field(100, description="Number of federated rounds")
    clients_per_round: int = Field(5, description="Number of clients sampled per round")
    local_epochs: int = Field(1, description="Number of local epochs per client")
    use_scaffold: bool = Field(False, description="Whether to enable SCAFFOLD")

class DpConfig(BaseModel):
    enabled: bool = Field(False, description="Enable Differential Privacy")
    noise_multiplier: float = Field(1.0, description="DP noise multiplier")
    clip_norm: float = Field(1.0, description="DP clipping norm")

class CryptoConfig(BaseModel):
    he_enabled: bool = Field(False, description="Enable homomorphic encryption")
    he_scheme: Optional[str] = Field(None, description="Name of HE scheme (e.g., TenSEAL)")

class ExperimentConfig(BaseModel):
    seed: int = Field(0, description="Global RNG seed")
    device: str = Field("cpu", description="Device for server orchestration")
    results_dir: str = Field("runs", description="Directory to store results and checkpoints")
    train: TrainConfig = TrainConfig()
    dp: DpConfig = DpConfig()
    crypto: CryptoConfig = CryptoConfig()

    class Config:
        arbitrary_types_allowed = True

# usage:
# from .config import ExperimentConfig
# cfg = ExperimentConfig.parse_file("configs/config_scaffold_dp.yaml")
