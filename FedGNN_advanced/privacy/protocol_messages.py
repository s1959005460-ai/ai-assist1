"""
protocol_messages.py

Message types used in the Bonawitz flow (in-memory objects for simulation).
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List


@dataclass
class RegisterClient:
    client_id: str
    param_shapes: Dict[str, Tuple[int, ...]]


@dataclass
class SharePackage:
    sender: str
    recipient: str
    share: Tuple[int, int]  # (idx, share_int)


@dataclass
class MaskedUpdate:
    sender: str
    masked_params: Dict[str, Any]  # tensors / numpy arrays depending on pipeline


@dataclass
class UnmaskRequest:
    requester: str  # typically server
    missing_clients: List[str]  # clients whose masks need reconstruction


@dataclass
class UnmaskShare:
    sender: str
    missing_client: str
    share: Tuple[int, int]  # part of Shamir share for the missing client's seed


@dataclass
class AggregatedResult:
    aggregated_params: Dict[str, Any]
