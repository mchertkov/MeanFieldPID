
from __future__ import annotations

from dataclasses import dataclass
import torch

from .protocol import PWCProtocol
from .time_domain import TimeDomain

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Numerically safe hyperbolic utilities (torch-only)
# -----------------------------------------------------------------------------

def _acoth(x: Tensor) -> Tensor:
    """acoth(x) for x>1 (caller should clamp to >1)."""
    return 0.5 * (torch.log(x + 1.0) - torch.log(x - 1.0))


def _coth(x: Tensor) -> Tensor:
    """Stable coth(x) with series fallback near 0 and exp form for large |x|."""
    ax = torch.abs(x)
    small = ax < 1e-6
    series = 1.0 / x + x / 3.0
    e2 = torch.exp(-2.0 * ax)
    coth_pos = (1.0 + e2) / (1.0 - e2)
    return torch.where(small, series, torch.sign(x) * coth_pos)


def _csch(x: Tensor) -> Tensor:
    """Stable csch(x)=1/sinh(x) with series fallback near 0."""
    ax = torch.abs(x)
    small = ax < 1e-6
    series = 1.0 / x - x / 6.0
    denom = -torch.expm1(-2.0 * ax)
    csch_pos = 2.0 * torch.exp(-ax) / denom
    return torch.where(small, series, torch.sign(x) * csch_pos)


# -----------------------------------------------------------------------------
# Continuous coefficients (Sec. 3.1.1 / App. B.1.1)
#   Continuous outputs only; no shifted r/s objects.
#   IMPORTANT: b_minus and c_minus are derived from a_minus via the backward
#   PWC propagation protocol (right-boundary anchors), matching the notes.
# -----------------------------------------------------------------------------

@dataclass
class ContinuousCoeffs:
    """Torch-only continuous-coefficient builder (Sec. 3.1.1 / App. B.1.1)."""

    beta: PWCProtocol  # (M,)
    nu: PWCProtocol    # (M,d)
    time_domain: TimeDomain | None = None

    def __post_init__(self):
        if self.beta.num_intervals != self.nu.num_intervals:
            raise ValueError("beta and nu must have the same number of intervals")
        if self.beta.time_domain.eps != self.nu.time_domain.eps:
            raise ValueError("beta and nu must share identical time_domain.eps")
        if not torch.allclose(self.beta.breaks, self.nu.breaks):
            raise ValueError("beta and nu must share identical breaks")
        if self.nu.values.ndim != 2:
            raise ValueError("nu.values must be (M,d)")

        self.breaks: Tensor = self.beta.breaks
        self.dtype = self.breaks.dtype
        self.device = self.breaks.device

        # Shared interior time policy
        self.time_domain = self.time_domain or self.beta.time_domain

        self.M = int(self.beta.num_intervals)
        self.d = int(self.nu.values.shape[1])

        self.Deltas = (self.breaks[1:] - self.breaks[:-1])             # (M,)
        self.beta_vals = self.beta.values.reshape(self.M)              # (M,)
        self.nu_vals = self.nu.values.reshape(self.M, self.d)          # (M,d)
        self.omega = torch.sqrt(torch.clamp(self.beta_vals, min=0.0))  # (M,)

        # Phases for the + branch (we keep the direct hyperbolic rep, since it
        # already matches legacy in your tests).
        self.phi_plus = self._compute_phi_plus()                       # (M,)
        self.theta_plus_bp = self._propagate_theta_plus_breakpoints()  # (M+1,d)
        self.theta_plus_1 = self.theta_plus_bp[-1]                     # (d,)
        self.a_plus_1 = self._a_plus_end_from_phi(self.M - 1, self.phi_plus[self.M - 1]).reshape(())

        # Backward (minus) scalar anchors for the PWC propagation protocol.
        # These are right-endpoint anchors for each interval i (i=0..M-2) at t_{i+1}.
        # Last interval (i=M-1) is evaluated by the terminal closed form.
        self._aR_minus, self._bR_minus, self._cR_minus = self._build_minus_right_anchors()

        # Interval constants phi_i^{(-)} inferred from right anchors a_{i+1}^{(-)} (non-terminal intervals),
        # and set to 0 on the terminal interval. This matches the sinh/cosh representation (B.19).
        self.phi_minus = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        if self.M > 1:
            i = torch.arange(self.M - 1, device=self.device)
            w = self.omega[i]
            aR = self._aR_minus[i]
            mask = w > 0
            x = torch.empty_like(w)
            x[mask] = aR[mask] / w[mask]
            # aR/omega = coth(phi) >= 1
            self.phi_minus[i[mask]] = _acoth(torch.clamp(x[mask], min=1.0 + 1e-9))
        # terminal interval i=M-1: phi=0

        # Theta_- breakpoints built using the continuous protocol (depends on a_minus/b_minus/c_minus). built using the continuous protocol (depends on b_minus/c_minus).
        self.theta_x_bp, self.theta_y_bp = self._propagate_theta_minus_breakpoints()

    # ------------------------------------------------------------------
    # interval location (delegated to protocol)
    # ------------------------------------------------------------------

    def locate(self, t: Tensor):
        """Return (idx, tau, Delta) for each t (torch tensors)."""
        return self.beta.locate(t)

    # ------------------------------------------------------------------
    # a_plus / theta_plus (direct hyperbolic rep)
    # ------------------------------------------------------------------

    def _a_plus_end_from_phi(self, i: int, phi_i: Tensor) -> Tensor:
        wi = self.omega[i]
        Di = self.Deltas[i]
        if float(wi.item()) == 0.0:
            return 1.0 / (Di + phi_i)
        return wi * _coth(wi * Di + phi_i)

    def _compute_phi_plus(self) -> Tensor:
        phi = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        phi[0] = 0.0
        for i in range(1, self.M):
            wi = self.omega[i]
            a_start = self._a_plus_end_from_phi(i - 1, phi[i - 1])
            if float(wi.item()) == 0.0:
                phi[i] = 1.0 / a_start
            else:
                x = torch.clamp(a_start / wi, min=1.0 + 1e-9)
                phi[i] = _acoth(x)
        return phi

    def a_plus(self, t: Tensor) -> Tensor:
        idx, tau, _ = self.locate(t)
        wi = self.omega[idx]
        phi = self.phi_plus[idx]
        z = wi * tau + phi
        a_h = wi * _coth(z)
        a_0 = 1.0 / (tau + phi)
        return torch.where(wi == 0, a_0, a_h)

    def _propagate_theta_plus_breakpoints(self) -> Tensor:
        th = torch.zeros(self.M + 1, self.d, dtype=self.dtype, device=self.device)
        th[0] = 0.0
        for i in range(self.M):
            wi = self.omega[i]
            Di = self.Deltas[i]
            phi = self.phi_plus[i]
            nu_i = self.nu_vals[i]
            th_i = th[i]
            if float(wi.item()) == 0.0:
                th[i + 1] = (phi / (Di + phi)) * th_i
            else:
                z = wi * Di + phi
                denom = torch.sinh(z)
                term1 = torch.sinh(phi) / denom * th_i
                term2 = (self.beta_vals[i] / wi) * ((torch.cosh(z) - torch.cosh(phi)) / denom) * nu_i
                th[i + 1] = term1 + term2
        return th

    def theta_plus(self, t: Tensor) -> Tensor:
        idx, tau, _ = self.locate(t)
        wi = self.omega[idx]
        phi = self.phi_plus[idx]
        th0 = self.theta_plus_bp[idx]
        nu = self.nu_vals[idx]
        if th0.ndim == 1:
            th0 = th0.unsqueeze(0)
        z = wi * tau + phi
        denom = torch.sinh(z).unsqueeze(-1)
        out_h = (
            torch.sinh(phi).unsqueeze(-1) / denom * th0
            + (self.beta_vals[idx] / wi).unsqueeze(-1)
            * ((torch.cosh(z) - torch.cosh(phi)).unsqueeze(-1) / denom)
            * nu
        )
        out_0 = (phi.unsqueeze(-1) / (tau.unsqueeze(-1) + phi.unsqueeze(-1))) * th0
        return torch.where((wi == 0).unsqueeze(-1), out_0, out_h)

    # ------------------------------------------------------------------
    # Backward scalar protocol (Sec. 3.1.1): anchor-based a-/b-/c-
    # ------------------------------------------------------------------

    def _last_interval_left_values(self) -> tuple[Tensor, Tensor, Tensor]:
        """Values at t_{M-1} for the terminal interval [t_{M-1},1] with phi=0."""
        i = self.M - 1
        wi = self.omega[i]
        Di = self.Deltas[i]
        if float(wi.item()) == 0.0:
            a = 1.0 / Di
            b = 1.0 / Di
            c = 1.0 / Di
        else:
            z = wi * Di
            a = wi * _coth(z)
            b = wi * _csch(z)
            c = wi * _coth(z)
        return a.reshape(()), b.reshape(()), c.reshape(())

    def _propagate_left_from_right(self, aR: Tensor, bR: Tensor, cR: Tensor, beta: Tensor, dur: Tensor):
        """One-step backward propagation across an interval of length dur."""
        r = torch.sqrt(torch.clamp(beta, min=0.0))
        if float(r.item()) == 0.0:
            aL = aR / (1.0 + aR * dur)
        else:
            th = torch.tanh(r * dur)
            aL = r * (aR + r * th) / (r + aR * th)

        # bL = bR * sqrt((aL^2 - beta)/(aR^2 - beta))
        # Within a single Riccati interval, a stays on the same side of sqrt(beta),
        # so (aL^2-beta) and (aR^2-beta) have the same sign and the ratio is >= 0.
        # Compute the ratio directly to handle the subcritical regime (a < sqrt(beta)).
        den = aR * aR - beta
        num = aL * aL - beta
        safe_den = torch.where(torch.abs(den) > 1e-30, den, torch.ones_like(den))
        ratio = num / safe_den
        bL = torch.where(torch.abs(den) > 1e-30, bR * torch.sqrt(torch.clamp(ratio, min=0.0)), bR)

        # cL = cR + (bR^2)/(beta - aR^2) * (aR - aL)
        denom = beta - aR * aR
        cL = torch.where(denom != 0, cR + (bR * bR) / denom * (aR - aL), cR)
        return aL.reshape(()), bL.reshape(()), cL.reshape(())

    def _build_minus_right_anchors(self):
        """Build (aR,bR,cR) at each right boundary t_{i+1} for i=0..M-2."""
        aR = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        bR = torch.zeros(self.M, dtype=self.dtype, device=self.device)
        cR = torch.zeros(self.M, dtype=self.dtype, device=self.device)

        # Start from the value at t_{M-1} coming from the terminal interval [t_{M-1},1].
        a0, b0, c0 = self._last_interval_left_values()

        # Walk backward over intervals i=M-2,...,0 storing right anchors at t_{i+1}.
        for i in range(self.M - 2, -1, -1):
            aR[i] = a0
            bR[i] = b0
            cR[i] = c0
            a0, b0, c0 = self._propagate_left_from_right(a0, b0, c0, self.beta_vals[i], self.Deltas[i])

        return aR, bR, cR

    def a_minus(self, t: Tensor) -> Tensor:
        idx, tau, Delta = self.locate(t)
        # tau is t - t_i; we need remaining-to-right: rem = Delta - tau
        rem = Delta - tau

        # Last interval uses terminal closed form with phi=0 and rem = 1-t.
        last = idx == (self.M - 1)

        # For non-last intervals: use right anchor at t_{i+1}.
        aR = self._aR_minus[idx]
        beta = self.beta_vals[idx]
        r = torch.sqrt(torch.clamp(beta, min=0.0))

        # r==0: a = aR/(1+aR*rem)
        a0 = aR / (1.0 + aR * rem)

        th = torch.tanh(r * rem)
        ah = r * (aR + r * th) / (r + aR * th)
        a_nonlast = torch.where(r == 0, a0, ah)

        # last interval formula
        wi = self.omega[idx]
        z = wi * rem
        a_last = torch.where(wi == 0, 1.0 / rem, wi * _coth(z))

        return torch.where(last, a_last, a_nonlast)

    def b_minus(self, t: Tensor) -> Tensor:
        idx, tau, Delta = self.locate(t)
        rem = Delta - tau
        last = idx == (self.M - 1)

        # non-last: b = bR * sqrt((a^2 - beta)/(aR^2 - beta))
        # Compute ratio directly to handle subcritical regime (a < sqrt(beta)).
        a = self.a_minus(t)
        aR = self._aR_minus[idx]
        bR = self._bR_minus[idx]
        beta = self.beta_vals[idx]
        den = aR * aR - beta
        num = a * a - beta
        safe_den = torch.where(torch.abs(den) > 1e-30, den, torch.ones_like(den))
        ratio = num / safe_den
        b_nonlast = torch.where(torch.abs(den) > 1e-30, bR * torch.sqrt(torch.clamp(ratio, min=0.0)), bR)

        # last: terminal closed form
        wi = self.omega[idx]
        z = wi * rem
        b_last = torch.where(wi == 0, 1.0 / rem, wi * _csch(z))

        return torch.where(last, b_last, b_nonlast)

    def c_minus(self, t: Tensor) -> Tensor:
        idx, tau, Delta = self.locate(t)
        rem = Delta - tau
        last = idx == (self.M - 1)

        a = self.a_minus(t)
        aR = self._aR_minus[idx]
        bR = self._bR_minus[idx]
        cR = self._cR_minus[idx]
        beta = self.beta_vals[idx]
        denom = beta - aR * aR
        c_nonlast = torch.where(denom != 0, cR + (bR * bR) / denom * (aR - a), cR)

        # last: terminal closed form
        wi = self.omega[idx]
        z = wi * rem
        c_last = torch.where(wi == 0, 1.0 / rem, wi * _coth(z))

        return torch.where(last, c_last, c_nonlast)


    # ------------------------------------------------------------------
    # theta_x/y minus (closed form App. B.1.1)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Centralized theta_y^- right-anchored update (single source of truth)
    # ------------------------------------------------------------------
    def _theta_y_minus_right_anchored(
                                     self,
                                     *,
                                     t: Tensor,
                                     tR: Tensor,
                                     aR: Tensor,
                                     bR: Tensor,
                                     txR: Tensor,
                                     tyR: Tensor,
                                     nu_i: Tensor,
                                     beta_i: Tensor) -> Tensor:
        r"""Right-anchored evaluation of :math:`\theta_y^{(-)}` consistent with the
        triangular ODE system (App. B.1.1):

            d/dt theta_y^{(-)}(t) = - b^{(-)}(t) * theta_x^{(-)}(t),

        with right endpoint anchor :math:`t_R=t_{i+1}^-`:

            theta_y(t) = theta_y(t_R) - \int_t^{t_R} b(s) theta_x(s) ds.

        Here theta_x(s) is evaluated in closed form using the already-implemented
        right-anchored expression (Eq. (B.31) for theta_x), i.e.

            theta_x(s) = (b(s)/bR) * txR + 1_{beta_i>0} * ( a(s) - (b(s)/bR) aR ) * nu_i.

        Implementation: fixed 8-point Gauss–Legendre quadrature per interval
        (vectorized), which is stable and sufficiently accurate for the smooth
        hyperbolic kernels used here.
        """
        # Shapes:
        #  - t,tR,aR,bR,beta_i: (N,)
        #  - txR,tyR,nu_i: (N,d)

        # Guard: if t==tR, integral is zero
        dt = (tR - t)
        if dt.ndim == 0:
            dt = dt.reshape(1)
        # 8-point Gauss-Legendre on [0,1]
        u = torch.tensor([
            0.0198550717512319,
            0.1016667612931866,
            0.2372337950418355,
            0.4082826787521751,
            0.5917173212478249,
            0.7627662049581645,
            0.8983332387068134,
            0.9801449282487681,
        ], dtype=self.dtype, device=self.device)  # (K,)
        w = torch.tensor([
            0.0506142681451881,
            0.1111905172266872,
            0.1568533229389436,
            0.1813418916891810,
            0.1813418916891810,
            0.1568533229389436,
            0.1111905172266872,
            0.0506142681451881,
        ], dtype=self.dtype, device=self.device)  # (K,)

        # Broadcast to (K,N)
        s = t.unsqueeze(0) + dt.unsqueeze(0) * u.unsqueeze(1)  # (K,N)

        # Evaluate scalar coeffs at s
        a_s = self.a_minus(s.reshape(-1)).reshape(s.shape)      # (K,N)
        b_s = self.b_minus(s.reshape(-1)).reshape(s.shape)      # (K,N)

        # theta_x(s) in closed form (right-anchored)
        mask = (beta_i != 0).to(self.dtype).unsqueeze(0).unsqueeze(-1)  # (1,N,1)
        b_ratio = (b_s / bR.unsqueeze(0)).unsqueeze(-1)                 # (K,N,1)
        aR_ = aR.unsqueeze(0).unsqueeze(-1)                             # (1,N,1)
        nu_ = nu_i.unsqueeze(0)                                         # (1,N,d)
        txR_ = txR.unsqueeze(0)                                         # (1,N,d)

        tx_s = b_ratio * txR_ + mask * (a_s.unsqueeze(-1) - b_ratio * aR_) * nu_  # (K,N,d)

        integrand = b_s.unsqueeze(-1) * tx_s                              # (K,N,d)
        # Quadrature: integral ≈ dt * sum_k w_k * integrand(t + dt*u_k)
        integral = dt.unsqueeze(-1) * torch.sum(w.unsqueeze(1).unsqueeze(-1) * integrand, dim=0)  # (N,d)

        return tyR - integral



    def _propagate_theta_minus_breakpoints(self) -> tuple[Tensor, Tensor]:
        """Build breakpoint values (theta_x^-, theta_y^-) by backward propagation.

        Notes (rev. MeanField_PID (67), App. B.1):
          - We do NOT anchor theta_y^- at t=1^- via (c-b)*nu evaluated at 1-eps.
          - Instead, we set theta_y^-(t_{M-1}) analytically using Eq. (B.29),
            and then propagate to earlier breakpoints using the right-anchored closed form Eq. (B.31),
            enforcing continuity (B.32) by construction.

        We keep the existing theta_x^- construction unchanged (it was already matching legacy),
        and only modify the theta_y^- breakpoint construction.
        """
        finfo = torch.finfo(self.dtype)
        eps = 16.0 * finfo.eps

        tx = torch.zeros(self.M + 1, self.d, dtype=self.dtype, device=self.device)
        ty = torch.zeros(self.M + 1, self.d, dtype=self.dtype, device=self.device)

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # theta_x^- : seed at t=1^- with the terminal limit theta_x^-(1)=0,
        # then propagate i=M-1,...,0 using the right-anchored closed form.
        # ------------------------------------------------------------------
        tx[-1] = 0.0

        for i in range(self.M - 1, -1, -1):
            tR = torch.clamp(self.breaks[i + 1] - eps, min=self.breaks[i] + eps)
            tL = torch.clamp(self.breaks[i] + eps, max=self.breaks[i + 1] - eps)

            aR = self.a_minus(tR.reshape(1))  # (1,)
            bR = self.b_minus(tR.reshape(1))
            aL = self.a_minus(tL.reshape(1))
            bL = self.b_minus(tL.reshape(1))

            txR = tx[i + 1].reshape(1, self.d)
            nu_i = self.nu_vals[i].reshape(1, self.d)
            beta_i = self.beta_vals[i].reshape(1)

            b_ratio = (bL / bR).unsqueeze(-1)  # (1,1)
            mask = (beta_i != 0).to(self.dtype).reshape(1, 1)

            txL = b_ratio * txR + mask * (aL.unsqueeze(-1) - b_ratio * aR.unsqueeze(-1)) * nu_i
            tx[i] = txL.reshape(self.d)

        # ------------------------------------------------------------------
        # theta_y^- :
        #   - terminal limit ty[M]=0 (theta_y^-(1)=0);
        #   - on the terminal interval, theta_y^-(t)=theta_x^-(t), hence at t_{M-1}:
        #         ty[M-1] = tx[M-1];
        #   - on earlier intervals, propagate by the right-anchored closed form (Eq. (B.33)).
        # ------------------------------------------------------------------
        ty[-1] = 0.0

        # terminal-interval left breakpoint t_{M-1}: theta_y^- = theta_x^-
        ty[self.M - 1] = tx[self.M - 1]

        # propagate earlier breakpoints using the closed form (Eq. (B.33))
        for i in range(self.M - 2, -1, -1):
            tR = torch.clamp(self.breaks[i + 1] - eps, min=self.breaks[i] + eps)
            tL = torch.clamp(self.breaks[i] + eps, max=self.breaks[i + 1] - eps)

            # Scalars at endpoints
            aR = self.a_minus(tR.reshape(1))  # (1,)
            bR = self.b_minus(tR.reshape(1))  # (1,)
            cR = self.c_minus(tR.reshape(1))  # (1,)

            bL = self.b_minus(tL.reshape(1))  # (1,)
            cL = self.c_minus(tL.reshape(1))  # (1,)

            # Right-endpoint anchors from breakpoint tables
            txR = tx[i + 1].reshape(1, self.d)
            tyR = ty[i + 1].reshape(1, self.d)

            nu_i = self.nu_vals[i].reshape(1, self.d)
            beta_i = self.beta_vals[i].reshape(1)
            mask = (beta_i != 0).to(self.dtype).reshape(1, 1)

            # Eq. (B.33): theta_y(t) = theta_y(tR) + ((cR-c)/bR) theta_x(tR)
            #             + ((bR-b) - aR (cR-c)/bR) * nu_i
            dc = (cR - cL) / bR  # (1,)
            nu_coef = (bR - bL) - aR * dc  # (1,)

            tyL = tyR + dc.unsqueeze(-1) * txR + (mask * nu_coef).unsqueeze(-1) * nu_i
            ty[i] = tyL.reshape(self.d)

        return tx, ty

    def _theta_minus_at(self, idx: Tensor, tau: Tensor, Delta: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate (theta_x^-, theta_y^-) inside an interval (App. B.1.1).

        Right-anchored construction with t_R=t_{i+1}^-:
          - theta_x^- uses the closed form already implemented (and validated against legacy).
          - theta_y^- uses the closed form in terms of (a^-,b^-,c^-) and right-endpoint anchors
                theta_y(t) = theta_y(tR) + ((c(tR)-c(t))/b(tR)) theta_x(tR) + ((b(tR)-b(t)) - a(tR)(c(tR)-c(t))/b(tR)) nu_i,
            with theta_x(s) evaluated in closed form (right-anchored).
          - On the terminal interval i=M-1, one has theta_y^-(t)=theta_x^-(t).
        """
        finfo = torch.finfo(self.dtype)
        eps = 16.0 * finfo.eps

        t = self.breaks[idx] + tau  # (N,)

        # Scalars at t
        a = self.a_minus(t)
        b = self.b_minus(t)

        # Interval constants
        nu_i = self.nu_vals[idx]     # (N,d)
        beta_i = self.beta_vals[idx] # (N,)
        mask = (beta_i != 0).to(self.dtype).unsqueeze(-1)  # (N,1)

        # Right endpoint (safe interior)
        tR = torch.clamp(self.breaks[idx + 1] - eps, min=self.breaks[idx] + eps)  # (N,)
        aR = self.a_minus(tR)  # (N,)
        bR = self.b_minus(tR)  # (N,)

        # Right-endpoint theta values (from breakpoint table)
        txR = self.theta_x_bp[idx + 1]  # (N,d)
        tyR = self.theta_y_bp[idx + 1]  # (N,d)

        # ----------------------------
        # theta_x^-
        # ----------------------------
        b_ratio = (b / bR).unsqueeze(-1)  # (N,1)
        out_tx = b_ratio * txR + mask * (a.unsqueeze(-1) - b_ratio * aR.unsqueeze(-1)) * nu_i

        # ----------------------------
        # theta_y^-
        # ----------------------------
        is_last = (idx == (self.M - 1))  # (N,)
        out_ty = torch.empty_like(out_tx)

        # Terminal interval: theta_y^- = theta_x^-
        out_ty = torch.where(is_last.unsqueeze(-1), out_tx, out_ty)

        # Non-terminal: integral update
        if torch.any(~is_last):
            # Closed form on non-terminal intervals (Eq. (B.33))
            c = self.c_minus(t)
            cR = self.c_minus(tR)
            dc = (cR - c) / bR  # (N,)
            nu_coef = (bR - b) - aR * dc  # (N,)

            out_ty_nonlast = tyR + dc.unsqueeze(-1) * txR + (mask.squeeze(-1) * nu_coef).unsqueeze(-1) * nu_i
            out_ty = torch.where((~is_last).unsqueeze(-1), out_ty_nonlast, out_ty)

        return out_tx, out_ty

    def theta_x_minus(self, t: Tensor) -> Tensor:
        idx, tau, Delta = self.locate(t)
        tx, _ = self._theta_minus_at(idx, tau, Delta)
        return tx

    def theta_y_minus(self, t: Tensor) -> Tensor:
        idx, tau, Delta = self.locate(t)
        _, ty = self._theta_minus_at(idx, tau, Delta)
        return ty