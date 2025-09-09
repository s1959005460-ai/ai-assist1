"""
topo_reg.py

Differentiable topology proxy regularizer with magic numbers moved to constants.
"""

from typing import Optional
import torch
import torch.nn as nn
from . import constants


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    prod = x @ x.t()
    dist2 = x2 + x2.t() - 2.0 * prod
    return torch.clamp(dist2, min=0.0)


class TopoReg(nn.Module):
    def __init__(self, margin: Optional[float] = None, k: Optional[int] = None, reduction: str = None):
        super().__init__()
        if margin is None:
            margin = constants.DEFAULTS.get("TOPO_MARGIN", 1.0)
        if k is None:
            k = constants.DEFAULTS.get("TOPO_K", 5)
        if reduction is None:
            reduction = constants.DEFAULTS.get("TOPO_REDUCTION", "mean")
        self.margin = float(margin)
        self.k = int(k)
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        dist2 = _pairwise_distances(embeddings)
        inf_mask = torch.eye(dist2.size(0), device=dist2.device).bool()
        dist2 = dist2 + (inf_mask * (1.0 / constants.DEFAULTS.get("LOG_TINY", 1e-300)))

        k = min(self.k, dist2.size(0) - 1) if dist2.size(0) > 1 else 0
        if k == 0:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

        knn_vals, _ = torch.topk(dist2, k=k, largest=False, sorted=False)
        knn_dists = torch.sqrt(torch.clamp(knn_vals, min=constants.DEFAULTS.get("EPS", 1e-12)))
        mean_knn = knn_dists.mean(dim=1)
        penalty = (mean_knn - self.margin).pow(2)

        if self.reduction == "mean":
            return penalty.mean()
        else:
            return penalty.sum()
