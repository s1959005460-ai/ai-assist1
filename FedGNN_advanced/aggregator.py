"""
aggregator.py

Robust aggregator with:
- torch.no_grad aggregation
- defensive handling for tiny total weight
- anomaly detector: client delta cosine similarity to mean delta
"""

from typing import Dict, List, Iterable, Any
import torch
from . import constants, logger
import numpy as np

class Aggregator:
    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_results: List[Dict[str, Any]],
        client_weights: Iterable[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client deltas into new global state.
        Defensive:
        - uses torch.no_grad()
        - ensures total weight >= AGG_MIN_TOTAL_WEIGHT (from constants)
        - returns detached, cloned tensors on configured device
        """
        client_weights = list(client_weights)
        if len(client_results) != len(client_weights):
            raise ValueError("client_results and client_weights must have same length")

        baseline = {k: v.to(self.device).detach().clone() for k, v in global_state.items()}

        with torch.no_grad():
            total_w = float(sum(client_weights)) if client_weights else 0.0
            if total_w < constants.DEFAULTS.get("AGG_MIN_TOTAL_WEIGHT", constants.AGG_MIN_TOTAL_WEIGHT):
                # defensive fallback: no update
                logger.secure_log("warning", "Total aggregation weight too small; skipping update", total_w=total_w)
                return {k: v.detach().clone().to(self.device) for k, v in baseline.items()}

            new_state: Dict[str, torch.Tensor] = {}
            for k, base_tensor in baseline.items():
                acc = torch.zeros_like(base_tensor, dtype=base_tensor.dtype, device=self.device)
                for i, client in enumerate(client_results):
                    delta = client.get("delta", {})
                    if k not in delta:
                        continue
                    d = delta[k].to(self.device).detach()
                    if d.dtype != acc.dtype:
                        d = d.to(acc.dtype)
                    acc.add_(client_weights[i] * d)
                # Weighted average
                update = acc.div(total_w)
                new_param = (base_tensor + update).detach().clone()
                new_state[k] = new_param
            return new_state


# ----------------
# anomaly detection utilities
# ----------------
def _flatten_state_to_vector(state: Dict[str, Any]) -> np.ndarray:
    """
    Flatten state dict (torch tensors or numpy arrays) into 1D numpy vector.
    Keys are processed in sorted order for determinism.
    """
    parts = []
    for k in sorted(state.keys()):
        v = state[k]
        if hasattr(v, "detach"):
            arr = v.detach().cpu().numpy().ravel()
        else:
            arr = np.array(v).ravel()
        parts.append(arr)
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts).astype(np.float64)


def client_delta_cosine_scores(global_state: Dict[str, Any], client_deltas: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    client_deltas: mapping client_id -> delta_state_dict
    returns mapping client_id -> cosine similarity between client delta and mean delta
    """
    vecs = {}
    for cid, state in client_deltas.items():
        vecs[cid] = _flatten_state_to_vector(state)
    if not vecs:
        return {}
    keys = list(vecs.keys())
    # compute mean delta
    mean = np.mean([vecs[k] for k in keys], axis=0)
    from numpy.linalg import norm
    scores = {}
    mean_norm = max(norm(mean), 1e-12)
    for k in keys:
        v = vecs[k]
        den = max(norm(v) * mean_norm, 1e-12)
        scores[k] = float(np.dot(v, mean) / den)
    return scores
