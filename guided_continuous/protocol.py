
from __future__ import annotations
from dataclasses import dataclass
import torch

from .time_domain import TimeDomain

Tensor = torch.Tensor

@dataclass(frozen=True)
class PWCProtocol:
    """Torch-only piecewise-constant protocol on [0,1]."""
    breaks: Tensor
    values: Tensor
    time_domain: TimeDomain = TimeDomain()

    def __post_init__(self):
        b = self.breaks
        if b.ndim != 1 or b.numel() < 2:
            raise ValueError("breaks must be 1D with length >= 2")
        if not torch.all(b[1:] > b[:-1]):
            raise ValueError("breaks must be strictly increasing")
        if torch.abs(b[0]) > 1e-12 or torch.abs(b[-1] - 1.0) > 1e-12:
            raise ValueError("breaks must start at 0 and end at 1")
        if self.values.shape[0] != b.numel() - 1:
            raise ValueError("values must have shape (len(breaks)-1, ...)")
        if not torch.is_floating_point(self.breaks) or not torch.is_floating_point(self.values):
            raise ValueError("breaks and values must be floating tensors")

    @property
    def num_intervals(self) -> int:
        return int(self.breaks.numel() - 1)

    def locate(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t = torch.as_tensor(t, dtype=self.breaks.dtype, device=self.breaks.device)
        # Enforce interior time domain policy (avoid t=0 and t=1).
        t_clamped = self.time_domain.clamp(t)
        idx = torch.searchsorted(self.breaks, t_clamped, right=True) - 1
        idx = torch.clamp(idx, 0, self.num_intervals - 1)
        b0 = self.breaks[idx]
        b1 = self.breaks[idx + 1]
        tau = t_clamped - b0
        Delta = b1 - b0
        return idx, tau, Delta

    def value_at(self, t: Tensor) -> Tensor:
        idx, _, _ = self.locate(t)
        return self.values[idx]
