from __future__ import annotations
from dataclasses import dataclass
import torch

Tensor = torch.Tensor

@dataclass
class GaussianMixture:
    """Torch-only Gaussian mixture target (continuous API)."""
    weights: Tensor  # (K,)
    means: Tensor    # (K,d)
    covs: Tensor     # (K,d,d)

    def __post_init__(self):
        if self.weights.ndim != 1:
            raise ValueError("weights must be (K,)")
        K = self.weights.numel()
        if self.means.shape[0] != K:
            raise ValueError("means must be (K,d)")
        if self.covs.shape[0] != K or self.covs.shape[1] != self.covs.shape[2]:
            raise ValueError("covs must be (K,d,d)")
        if self.covs.shape[1] != self.means.shape[1]:
            raise ValueError("means/covs dimension mismatch")
        self.weights = self.weights / torch.sum(self.weights)

    @property
    def K(self) -> int:
        return int(self.weights.numel())

    @property
    def d(self) -> int:
        return int(self.means.shape[1])
