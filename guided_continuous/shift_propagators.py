r"""Shift propagators for deterministic nonzero initial conditions.

Given a PWC protocol :math:`(\beta_t, \nu_t)`, the *base* vector coefficients
:math:`\theta^{(+)}_t, \theta^{(-)}_{x,t}, \theta^{(-)}_{y,t}` satisfy
linear ODEs driven by :math:`\beta_t \nu_t`.  When the initial condition is
shifted from 0 to :math:`z`, the protocol effectively becomes
:math:`(\beta_t, \nu_t - z)`, and by linearity of the ODEs the corrected
vector coefficients are:

.. math::
    \tilde\theta^{(+)}_t   &= \theta^{(+)}_t   - \lambda^{(+)}(t)\, z,\\
    \tilde\theta^{(-)}_{x,t} &= \theta^{(-)}_{x,t} + \lambda^{(-)}_{x}(t)\, z,\\
    \tilde\theta^{(-)}_{y,t} &= \theta^{(-)}_{y,t} + \lambda^{(-)}_{y}(t)\, z,

where the three *scalar* shift propagators :math:`\lambda^{(+)},
\lambda^{(-)}_x, \lambda^{(-)}_y` are independent of :math:`z` and satisfy
the same ODEs as the :math:`\theta` vectors but with the :math:`d`-vector
source :math:`\beta_t\nu_t` replaced by the scalar :math:`\beta_t`.

This module computes these propagators using the *same* PWC closed forms
already used for the :math:`\theta` coefficients, so the cost is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .coeffs_continuous import ContinuousCoeffs

Tensor = torch.Tensor


@dataclass
class ShiftPropagators:
    r"""Scalar shift propagators :math:`\lambda^{(+)}, \lambda^{(-)}_x,
    \lambda^{(-)}_y` for a given :class:`ContinuousCoeffs` object.

    Computed once; evaluation at arbitrary times is O(1) per query (closed
    forms within each PWC interval).
    """

    coeffs: ContinuousCoeffs

    def __post_init__(self):
        c = self.coeffs
        self.dtype = c.dtype
        self.device = c.device
        self.M = c.M

        # --- forward propagator λ⁺ ---
        self._lam_plus_bp = self._propagate_lambda_plus_breakpoints()   # (M+1,)
        self.lambda_plus_1: Tensor = self._lam_plus_bp[-1]              # scalar

        # --- backward propagators λ⁻ₓ, λ⁻ᵧ ---
        self._lam_x_bp, self._lam_y_bp = self._propagate_lambda_minus_breakpoints()  # (M+1,)

    # ------------------------------------------------------------------
    # λ⁺: forward propagation (same structure as θ⁺, with ν → 1)
    # ------------------------------------------------------------------

    def _propagate_lambda_plus_breakpoints(self) -> Tensor:
        """Build breakpoint values for λ⁺ by forward propagation.

        Same closed form as θ⁺ (Eq. B.29) but with νᵢ → 1 (scalar).
        """
        c = self.coeffs
        lam = torch.zeros(self.M + 1, dtype=self.dtype, device=self.device)
        lam[0] = 0.0                  # λ⁺(0) = 0

        for i in range(self.M):
            wi = c.omega[i]
            Di = c.Deltas[i]
            phi = c.phi_plus[i]
            lam_i = lam[i]

            if float(wi.item()) == 0.0:
                # β=0 interval: θ⁺ ODE reduces to λ̇⁺ = -a⁺λ⁺ with source 0
                # closed form: λ(t) = φ/(τ+φ) · λ(tᵢ)  (no source term when β=0)
                lam[i + 1] = (phi / (Di + phi)) * lam_i
            else:
                z = wi * Di + phi
                denom = torch.sinh(z)
                term1 = torch.sinh(phi) / denom * lam_i
                # Source term: (β/ω)(cosh(z)-cosh(φ))/sinh(z) · 1  (νᵢ→1)
                term2 = (c.beta_vals[i] / wi) * (torch.cosh(z) - torch.cosh(phi)) / denom
                lam[i + 1] = term1 + term2

        return lam

    def lambda_plus(self, t: Tensor) -> Tensor:
        """Evaluate λ⁺(t) at arbitrary times (scalar or batched)."""
        c = self.coeffs
        idx, tau, _ = c.locate(t)
        wi = c.omega[idx]
        phi = c.phi_plus[idx]
        lam0 = self._lam_plus_bp[idx]

        z = wi * tau + phi
        denom = torch.sinh(z)

        out_h = (
            torch.sinh(phi) / denom * lam0
            + (c.beta_vals[idx] / wi) * (torch.cosh(z) - torch.cosh(phi)) / denom
        )
        out_0 = (phi / (tau + phi)) * lam0
        return torch.where(wi == 0, out_0, out_h)

    # ------------------------------------------------------------------
    # λ⁻ₓ, λ⁻ᵧ: backward propagation (same structure as θ⁻ₓ, θ⁻ᵧ with ν → −1)
    # ------------------------------------------------------------------

    def _propagate_lambda_minus_breakpoints(self) -> tuple[Tensor, Tensor]:
        r"""Build breakpoint values for λ⁻ₓ and λ⁻ᵧ by backward propagation.

        λ⁻ₓ satisfies :math:`\dot\lambda^{(-)}_x = a^{(-)}\lambda^{(-)}_x + \beta`
        with :math:`\lambda^{(-)}_x(1)=0`.

        The right-anchored closed form for θ⁻ₓ is:
            θ⁻ₓ(t) = (b/bR)·θ⁻ₓ(tR) + [a(t) − (b/bR)aR]·νᵢ

        For λ⁻ₓ, the source is +β (vs −βν for θ⁻ₓ), so ν → −1:
            λ⁻ₓ(t) = (b/bR)·λ⁻ₓ(tR) − [a(t) − (b/bR)aR]

        Similarly for λ⁻ᵧ (same θ⁻ᵧ formula with νᵢ → −1).
        """
        c = self.coeffs
        finfo = torch.finfo(self.dtype)
        eps = 16.0 * finfo.eps

        lx = torch.zeros(self.M + 1, dtype=self.dtype, device=self.device)
        ly = torch.zeros(self.M + 1, dtype=self.dtype, device=self.device)

        # Terminal: λ⁻ₓ(1) = 0, λ⁻ᵧ(1) = 0
        lx[self.M] = 0.0
        ly[self.M] = 0.0

        # --- Terminal interval [t_{M-1}, 1): θ⁻ₓ(t) = ν·ω·tanh(ω(1-t)/2) ---
        # For λ⁻ₓ, ν → −1:  λ⁻ₓ(t) = −ω·tanh(ω(1-t)/2)
        i = self.M - 1
        wi = c.omega[i]
        Di = c.Deltas[i]
        if float(wi.item()) == 0.0:
            # β=0 terminal: θ⁻ₓ(t) = ν·(1-t)/2... limit of ω·tanh(ω·rem/2)
            # Actually for β=0, the ODE λ̇=a⁻λ+0 has λ=0 (no source), so:
            lx[i] = 0.0
        else:
            lx[i] = -wi * torch.tanh(wi * Di / 2.0)

        # On terminal interval, λ⁻ᵧ = λ⁻ₓ
        ly[i] = lx[i]

        # --- Earlier intervals: backward propagation ---
        for i in range(self.M - 2, -1, -1):
            tR = torch.clamp(c.breaks[i + 1] - eps, min=c.breaks[i] + eps)
            tL = torch.clamp(c.breaks[i] + eps, max=c.breaks[i + 1] - eps)

            aR = c.a_minus(tR.reshape(1)).squeeze()
            bR = c.b_minus(tR.reshape(1)).squeeze()
            aL = c.a_minus(tL.reshape(1)).squeeze()
            bL = c.b_minus(tL.reshape(1)).squeeze()
            cR_val = c.c_minus(tR.reshape(1)).squeeze()
            cL_val = c.c_minus(tL.reshape(1)).squeeze()

            beta_i = c.beta_vals[i]

            # λ⁻ₓ: right-anchored with ν → −1
            b_ratio = bL / bR
            bracket = aL - b_ratio * aR
            if float(beta_i.item()) != 0.0:
                lx[i] = b_ratio * lx[i + 1] - bracket    # νᵢ → −1
            else:
                lx[i] = b_ratio * lx[i + 1]              # no source when β=0

            # λ⁻ᵧ: right-anchored with ν → −1
            dc = (cR_val - cL_val) / bR
            nu_coef = (bR - bL) - aR * dc
            if float(beta_i.item()) != 0.0:
                ly[i] = ly[i + 1] + dc * lx[i + 1] - nu_coef    # νᵢ → −1
            else:
                ly[i] = ly[i + 1] + dc * lx[i + 1]

        return lx, ly

    def lambda_x_minus(self, t: Tensor) -> Tensor:
        """Evaluate λ⁻ₓ(t) at arbitrary times."""
        c = self.coeffs
        finfo = torch.finfo(self.dtype)
        eps = 16.0 * finfo.eps

        idx, tau, Delta = c.locate(t)
        t_abs = c.breaks[idx] + tau

        a = c.a_minus(t_abs)
        b = c.b_minus(t_abs)

        beta_i = c.beta_vals[idx]
        is_last = (idx == (self.M - 1))

        tR = torch.clamp(c.breaks[idx + 1] - eps, min=c.breaks[idx] + eps)
        aR = c.a_minus(tR)
        bR = c.b_minus(tR)

        lxR = self._lam_x_bp[idx + 1]

        b_ratio = b / bR
        bracket = a - b_ratio * aR
        mask = (beta_i != 0).to(self.dtype)

        out_nonlast = b_ratio * lxR - mask * bracket

        # Terminal interval: λ⁻ₓ(t) = −ω·tanh(ω(1-t)/2)
        wi = c.omega[idx]
        rem = 1.0 - t_abs
        out_last = torch.where(
            wi == 0,
            torch.zeros_like(t_abs),
            -wi * torch.tanh(wi * rem / 2.0),
        )

        return torch.where(is_last, out_last, out_nonlast)

    def lambda_y_minus(self, t: Tensor) -> Tensor:
        """Evaluate λ⁻ᵧ(t) at arbitrary times."""
        c = self.coeffs
        finfo = torch.finfo(self.dtype)
        eps = 16.0 * finfo.eps

        idx, tau, Delta = c.locate(t)
        t_abs = c.breaks[idx] + tau
        is_last = (idx == (self.M - 1))

        # Terminal interval: λ⁻ᵧ = λ⁻ₓ
        out_last = self.lambda_x_minus(t)

        # Non-terminal: right-anchored closed form
        tR = torch.clamp(c.breaks[idx + 1] - eps, min=c.breaks[idx] + eps)
        b_val = c.b_minus(t_abs)
        bR = c.b_minus(tR)
        c_val = c.c_minus(t_abs)
        cR_val = c.c_minus(tR)
        aR = c.a_minus(tR)

        beta_i = c.beta_vals[idx]
        mask = (beta_i != 0).to(self.dtype)

        lxR = self._lam_x_bp[idx + 1]
        lyR = self._lam_y_bp[idx + 1]

        dc = (cR_val - c_val) / bR
        nu_coef = (bR - b_val) - aR * dc

        out_nonlast = lyR + dc * lxR - mask * nu_coef    # νᵢ → −1

        return torch.where(is_last, out_last, out_nonlast)
