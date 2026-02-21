r"""Start-conditioned guided field for deterministic nonzero initial conditions.

For :math:`x_0 = z`, the controlled SDE is solved in shifted coordinates
:math:`\tilde x = x - z` with :math:`\tilde x_0 = 0`.  The score in shifted
coordinates is:

.. math::
    \tilde u_t^{(*)}(\tilde x;\,z)
    = b_t^{(-)}\!\bigl(\hat y_{\mathrm{shifted}} - \tilde\Upsilon_t(\tilde x)\bigr),

where the shift propagators from :mod:`.shift_propagators` provide the
:math:`z`-linear corrections to all vector coefficients.

**Absorption trick** (avoids per-particle target reconstruction):

Shifting the target means :math:`m_k \to m_k - z` and the probe mean
:math:`\mu \to \tilde\mu` is algebraically equivalent to keeping the original
target means and evaluating :math:`\hat y` at an *effective* probe mean

.. math::
    \mu_{\mathrm{eff}} = \tilde\mu(\tilde x) + z,

then subtracting :math:`z` from the result:
:math:`\hat y_{\mathrm{shifted}} = \hat y_{\mathrm{standard}}(\mu_{\mathrm{eff}}) - z`.

This avoids any per-particle modification of the target GMM and allows full
batch vectorization over diverse :math:`z` values.
"""

from __future__ import annotations

import math
import torch

from .guided_field import GuidedField, _as_batch_time
from .shift_propagators import ShiftPropagators
from .gaussian_mixture import GaussianMixture

Tensor = torch.Tensor


def _yhat_from_mu(
    mu: Tensor,
    Kt: Tensor,
    target: GaussianMixture,
) -> Tensor:
    r"""Compute :math:`\hat y(t;x)` given an externally provided probe mean.

    This is a standalone version of :meth:`GuidedField.yhat` that accepts
    ``mu`` as an argument rather than computing it internally.  This allows
    the shifted field to inject the corrected probe mean (including the
    absorption of the target-mean shift).

    Args:
        mu:     (B, d) probe mean
        Kt:     (B,)   probe precision
        target: GaussianMixture with original (unshifted) means/covs/weights
    Returns:
        yhat:   (B, d)
    """
    B, d = mu.shape
    dtype, device = mu.dtype, mu.device
    I = torch.eye(d, dtype=dtype, device=device).reshape(1, d, d)

    logw_list = []
    mtilde_list = []

    for k in range(target.K):
        Sig = target.covs[k].to(dtype=dtype, device=device).reshape(1, d, d)
        mk = target.means[k].to(dtype=dtype, device=device).reshape(1, d)

        # Evaluate N(mu; m_k, Sigma_k + I/Kt)
        cov_eval = Sig + (1.0 / Kt).reshape(B, 1, 1) * I
        Lc = torch.linalg.cholesky(cov_eval)
        diff = mu - mk
        sol = torch.cholesky_solve(diff.unsqueeze(-1), Lc).squeeze(-1)
        quad = torch.sum(diff * sol, dim=1)
        logdet = 2.0 * torch.sum(
            torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)), dim=1
        )
        log2pi = math.log(2.0 * math.pi)
        logNk = -0.5 * (quad + logdet + d * log2pi)
        logw_list.append(
            torch.log(target.weights[k].to(dtype=dtype, device=device)) + logNk
        )

        # Posterior mean: (Sig^{-1} + Kt I)^{-1} (Sig^{-1} m_k + Kt mu)
        Sig_inv = torch.linalg.inv(Sig)
        Prec = Sig_inv + Kt.reshape(B, 1, 1) * I
        Lp = torch.linalg.cholesky(Prec)
        rhs = (
            (Sig_inv @ mk.unsqueeze(-1)).squeeze(-1).expand(B, d)
            + Kt.reshape(B, 1) * mu
        )
        mt = torch.cholesky_solve(rhs.unsqueeze(-1), Lp).squeeze(-1)
        mtilde_list.append(mt)

    logw = torch.stack(logw_list, dim=1)          # (B, K)
    mtilde = torch.stack(mtilde_list, dim=1)      # (B, K, d)
    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
    wts = torch.exp(logw).unsqueeze(-1)           # (B, K, 1)
    return torch.sum(wts * mtilde, dim=1)         # (B, d)


class ShiftedField:
    r"""Score field for :math:`x_0 = z` via shift propagators.

    Usage::

        sp = ShiftPropagators(coeffs)
        sf = ShiftedField(field, sp)

        # SDE in shifted coords:  d x̃ = sf.u_star(t, x̃, z) dt + dW,  x̃₀=0
        # Physical coords:         X_t = x̃_t + z

    All methods accept batched ``z`` of shape ``(B, d)``, enabling vectorized
    simulation over diverse initial conditions.
    """

    def __init__(self, field: GuidedField, propagators: ShiftPropagators):
        self.field = field
        self.prop = propagators
        self.coeffs = field.coeffs
        self.target = field.target
        self.time_domain = field.time_domain

    # ------------------------------------------------------------------
    # Core: shifted score u*_t(x̃; z)
    # ------------------------------------------------------------------

    def u_star(self, t: Tensor, x_tilde: Tensor, z: Tensor) -> Tensor:
        r"""Shifted optimal control in shifted coordinates.

        Args:
            t:       scalar or (B,)
            x_tilde: (B, d) — current state in shifted coords (x̃₀ = 0)
            z:       (B, d) — deterministic initial offset(s)

        Returns:
            drift:   (B, d)
        """
        if x_tilde.ndim != 2 or z.ndim != 2:
            raise ValueError("x_tilde and z must be (B, d)")
        B, d = x_tilde.shape
        tt = _as_batch_time(t, B, x_tilde.dtype, x_tilde.device)
        tt = self.time_domain.clamp(tt)

        # --- base quantities (z-independent) ---
        Kt = self.field.K(tt)                          # (B,)
        mu_base = self.field.mu(tt, x_tilde)           # (B, d)
        Upsilon_base = self.field.Upsilon(tt, x_tilde) # (B, d)
        b = self.coeffs.b_minus(tt)                    # (B,)

        # --- shift propagator values ---
        lam_plus_1 = self.prop.lambda_plus_1           # scalar
        lam_x = self.prop.lambda_x_minus(tt)           # (B,)
        lam_y = self.prop.lambda_y_minus(tt)           # (B,)

        # --- corrected μ (absorption trick) ---
        # μ̃ = μ_base + (λ⁻ᵧ + λ⁺₁)/K_t · z
        # μ_eff = μ̃ + z = μ_base + [(λ⁻ᵧ + λ⁺₁)/K_t + 1] · z
        delta_coeff = (lam_y + lam_plus_1) / Kt + 1.0  # (B,)
        mu_eff = mu_base + delta_coeff.unsqueeze(-1) * z  # (B, d)

        # --- ŷ via absorption trick ---
        # ŷ_shifted = ŷ_standard(μ_eff, original target) − z
        yhat_std = _yhat_from_mu(mu_eff, Kt, self.target)  # (B, d)
        yhat_shifted = yhat_std - z                          # (B, d)

        # --- corrected Υ ---
        # Υ̃ = Υ_base − (λ⁻ₓ / b⁻) · z
        Upsilon_shifted = Upsilon_base - (lam_x / b).unsqueeze(-1) * z

        # --- score ---
        return b.unsqueeze(-1) * (yhat_shifted - Upsilon_shifted)
