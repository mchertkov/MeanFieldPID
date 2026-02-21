from __future__ import annotations

from dataclasses import dataclass
import torch

Tensor = torch.Tensor

@dataclass(frozen=True)
class TimeDomain:
    """Interior time domain policy for all evaluations.

    We avoid evaluating analytic expressions at t=0 or t=1 due to endpoint
    singular asymptotics (e.g., ~1/t or ~1/(1-t)). The allowed domain is
    [eps, 1-eps], enforced by clamp().
    """
    eps: float = 1e-4

    def clamp(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t)
        lo = torch.as_tensor(self.eps, dtype=t.dtype, device=t.device)
        hi = torch.as_tensor(1.0 - self.eps, dtype=t.dtype, device=t.device)
        return torch.clamp(t, lo, hi)
