from __future__ import annotations

from dataclasses import dataclass
import torch

from .time_domain import TimeDomain

Tensor = torch.Tensor


@dataclass
class EulerMaruyamaResult:
    times: Tensor          # (T,)
    traj: Tensor           # (T,B,d)
    dt: Tensor             # (T-1,) variable dt allowed


def _build_break_aligned_grid(
    *,
    n_steps: int,
    breaks: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    time_domain: TimeDomain,
) -> Tensor:
    """
    Build a time grid on [eps, 1-eps] that contains (clamped) breakpoints.
    Steps are allocated per interval proportionally to interval length.
    """
    eps = float(time_domain.eps)
    lo = eps
    hi = 1.0 - eps

    b = torch.as_tensor(breaks, dtype=dtype, device=device).flatten()
    # keep, sort, unique
    b = torch.unique(torch.sort(b).values)

    # clamp breaks to [lo, hi]
    b = torch.clamp(b, lo, hi)

    # ensure endpoints are present
    if b.numel() == 0 or float(b[0].item()) > lo + 1e-15:
        b = torch.cat([torch.tensor([lo], dtype=dtype, device=device), b])
    else:
        b[0] = torch.tensor(lo, dtype=dtype, device=device)
    if float(b[-1].item()) < hi - 1e-15:
        b = torch.cat([b, torch.tensor([hi], dtype=dtype, device=device)])
    else:
        b[-1] = torch.tensor(hi, dtype=dtype, device=device)

    # interval lengths
    lengths = (b[1:] - b[:-1]).clamp_min(0.0)  # (K-1,)
    L = float(lengths.sum().item())
    if L <= 0.0:
        # degenerate: fall back
        return torch.linspace(lo, hi, n_steps + 1, dtype=dtype, device=device)

    # allocate integer steps per interval proportional to length, at least 1
    raw = lengths / L * float(n_steps)
    n = torch.floor(raw).to(torch.long)
    n = torch.clamp(n, min=1)

    # adjust to match total n_steps
    deficit = int(n_steps - int(n.sum().item()))
    if deficit > 0:
        # add steps to the largest fractional parts
        frac = (raw - torch.floor(raw))
        idx = torch.argsort(frac, descending=True)
        for k in idx[:deficit]:
            n[k] += 1
    elif deficit < 0:
        # remove steps from intervals with largest n, but keep >=1
        idx = torch.argsort(n, descending=True)
        k = 0
        while deficit < 0 and k < idx.numel():
            ii = idx[k].item()
            if n[ii] > 1:
                n[ii] -= 1
                deficit += 1
            else:
                k += 1

    # build piecewise grid; include breakpoints exactly, avoid duplicates
    pieces = []
    for k in range(b.numel() - 1):
        t0 = b[k]
        t1 = b[k + 1]
        nk = int(n[k].item())
        # nk steps means nk+1 points on this interval; we will drop the last point except for final interval
        tt = torch.linspace(t0, t1, nk + 1, dtype=dtype, device=device)
        if k < b.numel() - 2:
            tt = tt[:-1]
        pieces.append(tt)

    return torch.cat(pieces, dim=0)


def _make_break_set(
    breaks: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    time_domain: TimeDomain,
) -> Tensor:
    """Unique sorted breakpoints clamped to [eps, 1-eps]."""
    eps = float(time_domain.eps)
    b = torch.as_tensor(breaks, dtype=dtype, device=device).flatten()
    b = torch.clamp(b, eps, 1.0 - eps)
    return torch.unique(torch.sort(b).values)


def _time_eval_right_limit(
    t: Tensor,
    bset: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    post_jump_eps: float,
) -> Tensor:
    """
    If t hits a breakpoint (within floating tolerance), return t + post_jump_eps
    to pick the right-limit regime; else return t.
    """
    tol = 10.0 * torch.finfo(dtype).eps
    if torch.any(torch.abs(bset - t) <= tol):
        return t + torch.as_tensor(post_jump_eps, dtype=dtype, device=device)
    return t


def euler_maruyama_guided(
    u_star,
    *,
    B: int,
    d: int,
    n_steps: int = 2000,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    seed: int = 0,
    x0_zero: bool = True,
    time_domain: TimeDomain = TimeDomain(),
    breaks: Tensor | None = None,
    post_jump_eps: float = 1e-12,
) -> EulerMaruyamaResult:
    r"""Simulate dX_t = u*_t(X_t) dt + dW_t on [0,1] via Euler--Maruyama.

    If breaks is provided, use a breakpoint-aligned time grid and evaluate u_star
    at exact break times using a right-limit convention t -> t + post_jump_eps.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # time grid
    if breaks is None:
        eps = float(time_domain.eps)
        times = torch.linspace(eps, 1.0 - eps, n_steps + 1, dtype=dtype, device=device)
        bset = None
    else:
        times = _build_break_aligned_grid(
            n_steps=n_steps,
            breaks=breaks,
            dtype=dtype,
            device=device,
            time_domain=time_domain,
        )
        bset = _make_break_set(breaks, dtype=dtype, device=device, time_domain=time_domain)

    dt_vec = times[1:] - times[:-1]  # (T-1,)

    # init
    if x0_zero:
        x = torch.zeros((B, d), dtype=dtype, device=device)
    else:
        x = torch.randn((B, d), dtype=dtype, device=device, generator=gen)

    traj = torch.empty((times.numel(), B, d), dtype=dtype, device=device)
    traj[0] = x

    for i in range(times.numel() - 1):
        t = times[i]
        dt = dt_vec[i]
        sqrt_dt = torch.sqrt(dt)

        t_eval = t
        if bset is not None:
            t_eval = _time_eval_right_limit(t, bset, dtype=dtype, device=device, post_jump_eps=post_jump_eps)

        drift = u_star(t_eval, x)  # (B,d)
        noise = torch.randn((B, d), dtype=dtype, device=device, generator=gen)
        x = x + drift * dt + sqrt_dt * noise
        traj[i + 1] = x

    return EulerMaruyamaResult(times=times, traj=traj, dt=dt_vec)


def heun_guided(
    u_star,
    *,
    B: int,
    d: int,
    n_steps: int = 2000,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    seed: int = 0,
    x0_zero: bool = True,
    time_domain: TimeDomain = TimeDomain(),
    breaks: Tensor | None = None,
    post_jump_eps: float = 1e-12,
) -> EulerMaruyamaResult:
    r"""Simulate dX_t = u*_t(X_t) dt + dW_t on [0,1] via stochastic Heun.

    Additive noise (sigma=1) predictor-corrector:
        ξ ~ N(0,I)
        x_pred = x + u(t,x) dt + sqrt(dt) ξ
        x_next = x + 0.5*(u(t,x) + u(t+dt, x_pred)) dt + sqrt(dt) ξ

    If breaks is provided, uses breakpoint-aligned grid and right-limit evaluation
    at exact break times for BOTH t and t+dt.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # time grid
    if breaks is None:
        eps = float(time_domain.eps)
        times = torch.linspace(eps, 1.0 - eps, n_steps + 1, dtype=dtype, device=device)
        bset = None
    else:
        times = _build_break_aligned_grid(
            n_steps=n_steps,
            breaks=breaks,
            dtype=dtype,
            device=device,
            time_domain=time_domain,
        )
        bset = _make_break_set(breaks, dtype=dtype, device=device, time_domain=time_domain)

    dt_vec = times[1:] - times[:-1]  # (T-1,)

    # init
    if x0_zero:
        x = torch.zeros((B, d), dtype=dtype, device=device)
    else:
        x = torch.randn((B, d), dtype=dtype, device=device, generator=gen)

    traj = torch.empty((times.numel(), B, d), dtype=dtype, device=device)
    traj[0] = x

    for i in range(times.numel() - 1):
        t = times[i]
        t_next = times[i + 1]
        dt = dt_vec[i]
        sqrt_dt = torch.sqrt(dt)

        # shared noise increment
        noise = torch.randn((B, d), dtype=dtype, device=device, generator=gen)

        # drift at current time (right-limit at breakpoints)
        t_eval = t
        if bset is not None:
            t_eval = _time_eval_right_limit(t, bset, dtype=dtype, device=device, post_jump_eps=post_jump_eps)

        drift = u_star(t_eval, x)  # (B,d)

        # predictor
        x_pred = x + drift * dt + sqrt_dt * noise

        # drift at next time (right-limit if t_next is a breakpoint)
        t_next_eval = t_next
        if bset is not None:
            t_next_eval = _time_eval_right_limit(t_next, bset, dtype=dtype, device=device, post_jump_eps=post_jump_eps)

        drift_pred = u_star(t_next_eval, x_pred)  # (B,d)

        # corrector
        x = x + 0.5 * (drift + drift_pred) * dt + sqrt_dt * noise
        traj[i + 1] = x

    return EulerMaruyamaResult(times=times, traj=traj, dt=dt_vec)
