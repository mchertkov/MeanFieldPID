from __future__ import annotations

from dataclasses import dataclass
import torch

from .coeffs_continuous import ContinuousCoeffs
from .gaussian_mixture import GaussianMixture
from .time_domain import TimeDomain

Tensor = torch.Tensor


def _as_batch_time(t: Tensor, B: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return t as shape (B,) on the requested dtype/device."""
    t = torch.as_tensor(t, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.expand(B)
    if t.ndim == 1 and t.numel() == 1:
        return t.expand(B)
    if t.ndim == 1 and t.numel() == B:
        return t
    raise ValueError(f"t must be scalar or (B,), got shape {tuple(t.shape)} with B={B}")


@dataclass
class GuidedField:
    """Analytic guided quantities u*_t(x), p*_t(x), and derived maps.

    All computations are torch-only (CUDA/autograd friendly).
    """

    coeffs: ContinuousCoeffs
    target: GaussianMixture
    time_domain: TimeDomain | None = None

    def __post_init__(self):
        self.time_domain = self.time_domain or self.coeffs.time_domain or self.coeffs.beta.time_domain

    # ---------------------------------------------------------------------
    # Target-independent derived continuous coefficients (B.9, B.11)
    # ---------------------------------------------------------------------

    def K(self, t: Tensor) -> Tensor:
        t = self.time_domain.clamp(torch.as_tensor(t, dtype=self.coeffs.dtype, device=self.coeffs.device))
        return self.coeffs.c_minus(t) - self.coeffs.a_plus_1

    def alpha(self, t: Tensor) -> Tensor:
        t = self.time_domain.clamp(torch.as_tensor(t, dtype=self.coeffs.dtype, device=self.coeffs.device))
        return self.coeffs.b_minus(t) / self.K(t)

    def dbar(self, t: Tensor) -> Tensor:
        t = self.time_domain.clamp(torch.as_tensor(t, dtype=self.coeffs.dtype, device=self.coeffs.device))
        # (B,d): theta_y_minus(t) is (B,d) and theta_plus_1 is (d,)
        return (self.coeffs.theta_y_minus(t) - self.coeffs.theta_plus_1.unsqueeze(0)) / self.K(t).unsqueeze(-1)

    def mu(self, t: Tensor, x: Tensor) -> Tensor:
        # mu_t(x) = alpha_t x + dbar_t
        B, _ = x.shape
        tt = _as_batch_time(t, B, x.dtype, x.device)
        tt = self.time_domain.clamp(tt)
        tt = self.time_domain.clamp(tt)
        return self.alpha(tt).unsqueeze(-1) * x + self.dbar(tt)

    def Upsilon(self, t: Tensor, x: Tensor) -> Tensor:
        # Upsilon_t(x) = (-theta_x_minus(t) + a_minus(t) x) / b_minus(t)
        if x.ndim != 2:
            raise ValueError("x must be (B,d)")
        B, _ = x.shape
        tt = _as_batch_time(t, B, x.dtype, x.device)
        tt = self.time_domain.clamp(tt)
        tt = self.time_domain.clamp(tt)
        num = - self.coeffs.theta_x_minus(tt) + self.coeffs.a_minus(tt).unsqueeze(-1) * x
        return num / self.coeffs.b_minus(tt).unsqueeze(-1)

    # ---------------------------------------------------------------------
    # Target-dependent objects: yhat and u_star (A.24--A.26 / B.8--B.11)
    # ---------------------------------------------------------------------

    def yhat(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute \hat y(t;x) for batched x.

        Args:
            t: scalar or (B,)
            x: (B,d)
        Returns:
            (B,d)
        """
        if x.ndim != 2:
            raise ValueError("x must be (B,d)")
        B, d = x.shape
        tt = _as_batch_time(t, B, x.dtype, x.device)
        tt = self.time_domain.clamp(tt)
        tt = self.time_domain.clamp(tt)

        Kt = self.K(tt)  # (B,)
        mu = self.mu(tt, x)  # (B,d)
        I = torch.eye(d, dtype=x.dtype, device=x.device).reshape(1, d, d)

        logw_list = []
        mtilde_list = []

        for k in range(self.target.K):
            Sig = self.target.covs[k].reshape(1, d, d)

            # bar-pi_k(t;x) uses N(mu_t(x); m_k, Sigma_k + I/Kt)
            cov_eval = Sig + (1.0 / Kt).reshape(B, 1, 1) * I
            Lc = torch.linalg.cholesky(cov_eval)
            diff = mu - self.target.means[k].reshape(1, d)
            sol = torch.cholesky_solve(diff.unsqueeze(-1), Lc).squeeze(-1)
            quad = torch.sum(diff * sol, dim=1)
            logdet = 2.0 * torch.sum(torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)), dim=1)

            log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
            logNk = -0.5 * (quad + logdet + d * log2pi)
            logw_list.append(torch.log(self.target.weights[k]) + logNk)

            # tilde m_k(t;x) = (Sigma^{-1} + Kt I)^{-1} (Sigma^{-1} m_k + Kt mu)
            Sig_inv = torch.linalg.inv(self.target.covs[k]).reshape(1, d, d)
            Prec = Sig_inv + Kt.reshape(B, 1, 1) * I
            Lp = torch.linalg.cholesky(Prec)
            rhs = (Sig_inv @ self.target.means[k].reshape(1, d, 1)).squeeze(-1) + Kt.reshape(B, 1) * mu
            mt = torch.cholesky_solve(rhs.unsqueeze(-1), Lp).squeeze(-1)
            mtilde_list.append(mt)

        logw = torch.stack(logw_list, dim=1)        # (B,K)
        mtilde = torch.stack(mtilde_list, dim=1)    # (B,K,d)
        logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
        wts = torch.exp(logw).unsqueeze(-1)         # (B,K,1)
        return torch.sum(wts * mtilde, dim=1)       # (B,d)

    def u_star(self, t: Tensor, x: Tensor) -> Tensor:
        """Optimal control field u*_t(x) = b^-_t ( \hat y(t;x) - Upsilon_t(x) )."""
        if x.ndim != 2:
            raise ValueError("x must be (B,d)")
        B, _ = x.shape
        tt = _as_batch_time(t, B, x.dtype, x.device)
        tt = self.time_domain.clamp(tt)
        tt = self.time_domain.clamp(tt)
        b = self.coeffs.b_minus(tt).unsqueeze(-1)
        return b * (self.yhat(tt, x) - self.Upsilon(tt, x))

    # ---------------------------------------------------------------------
    # Explicit optimal marginal density p*_t(x) (B.12--B.14)
    # ---------------------------------------------------------------------

    def _Sk(self, Kt: Tensor, d: int, Sig: Tensor) -> Tensor:
        # Sk(t) = Sigma_k + I/Kt
        I = torch.eye(d, dtype=Sig.dtype, device=Sig.device)
        return Sig + (1.0 / Kt).reshape(-1, 1, 1) * I  # (B,d,d) if Kt is (B,)

    def _Mk_hk(self, tt: Tensor, x: Tensor | None = None):
        """Compute Mk(t) and hk(t) for each component k.

        Returns:
            Mk: (B,K,d,d)
            hk: (B,K,d)
            Sk: (B,K,d,d)
            qk: (B,K) where qk = (m_k-dbar)^T Sk^{-1} (m_k-dbar)
        """
        # We evaluate at batch times; if x is provided we use its dtype/device.
        if x is None:
            dtype = self.coeffs.breaks.dtype
            device = self.coeffs.breaks.device
            B = tt.numel() if tt.ndim == 1 else 1
            d = self.target.d
        else:
            B, d = x.shape
            dtype, device = x.dtype, x.device

        tt = torch.as_tensor(tt, dtype=dtype, device=device)
        if tt.ndim == 0:
            tt = tt.expand(B)
        elif tt.ndim == 1 and tt.numel() == 1:
            tt = tt.expand(B)

        Kt = self.K(tt)                     # (B,)
        atp = self.coeffs.a_plus(tt)        # (B,)
        atm = self.coeffs.a_minus(tt)       # (B,)
        bm = self.coeffs.b_minus(tt)        # (B,)
        alph = self.alpha(tt)               # (B,)
        dbar = self.dbar(tt)                # (B,d)
        thp = self.coeffs.theta_plus(tt)    # (B,d)
        thx = self.coeffs.theta_x_minus(tt) # (B,d)

        I = torch.eye(d, dtype=dtype, device=device).reshape(1, 1, d, d)

        Mk_list = []
        hk_list = []
        Sk_list = []
        qk_list = []

        base_scalar = (atp + atm - (bm * bm) / Kt)  # (B,)
        base = base_scalar.reshape(B, 1, 1, 1) * I  # (B,1,d,d)

        for k in range(self.target.K):
            Sig = self.target.covs[k].to(dtype=dtype, device=device).reshape(1, d, d)
            Sk = Sig + (1.0 / Kt).reshape(B, 1, 1) * torch.eye(d, dtype=dtype, device=device).reshape(1, d, d)  # (B,d,d)

            Ls = torch.linalg.cholesky(Sk)                  # (B,d,d)
            Sk_inv = torch.cholesky_solve(torch.eye(d, dtype=dtype, device=device).reshape(1, d, d).expand(B, d, d), Ls)

            Mk = base.squeeze(1) + (alph * alph).reshape(B, 1, 1) * Sk_inv  # (B,d,d)

            mk = self.target.means[k].to(dtype=dtype, device=device).reshape(1, d)
            diff = (mk - dbar)  # (B,d)
            # qk = diff^T Sk^{-1} diff
            tmp = torch.cholesky_solve(diff.unsqueeze(-1), Ls).squeeze(-1)  # (B,d)
            qk = torch.sum(diff * tmp, dim=1)  # (B,)

            hk = thp + thx + bm.unsqueeze(-1) * dbar + alph.unsqueeze(-1) * (Sk_inv @ (mk - dbar).unsqueeze(-1)).squeeze(-1)

            Mk_list.append(Mk)
            hk_list.append(hk)
            Sk_list.append(Sk)
            qk_list.append(qk)

        Mk = torch.stack(Mk_list, dim=1)  # (B,K,d,d)
        hk = torch.stack(hk_list, dim=1)  # (B,K,d)
        Sk = torch.stack(Sk_list, dim=1)  # (B,K,d,d)
        qk = torch.stack(qk_list, dim=1)  # (B,K)
        return Mk, hk, Sk, qk

    def log_p_star(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute log p*_t(x) using Eq. (B.12) in a stable log-sum-exp form.

        Args:
            t: scalar or (B,)
            x: (B,d)
        Returns:
            log p*_t(x): (B,)
        """
        if x.ndim != 2:
            raise ValueError("x must be (B,d)")
        B, d = x.shape
        tt = _as_batch_time(t, B, x.dtype, x.device)
        tt = self.time_domain.clamp(tt)
        tt = self.time_domain.clamp(tt)

        Mk, hk, Sk, qk = self._Mk_hk(tt, x=x)  # (B,K,*,*)
        # log |Sk|
        logdet_S = torch.logdet(Sk)            # (B,K)
        # log component weight terms
        logpi = torch.log(self.target.weights.to(dtype=x.dtype, device=x.device)).reshape(1, -1).expand(B, -1)  # (B,K)

        # quadratic form x^T Mk x
        x_col = x.unsqueeze(1).unsqueeze(-1)  # (B,1,d,1)
        Mx = torch.matmul(Mk, x_col).squeeze(-1)  # (B,K,d)  # (B,K,d)
        quad = torch.sum(x.unsqueeze(1) * Mx, dim=2)  # (B,K)
        lin = torch.sum(hk * x.unsqueeze(1), dim=2)    # (B,K)

        log_num_terms = logpi - 0.5 * logdet_S - 0.5 * quad + lin - 0.5 * qk  # (B,K)
        log_num = torch.logsumexp(log_num_terms, dim=1)  # (B,)

        # denominator Z(t): sum_l pi_l |Sl|^{-1/2} |Ml|^{-1/2} exp( 1/2 h^T M^{-1} h - 1/2 q )
        logdet_M = torch.logdet(Mk)            # (B,K)
        # solve M^{-1} h for each k
        # (B,K,d,d) and (B,K,d)
        v = torch.linalg.solve(Mk, hk.unsqueeze(-1)).squeeze(-1)  # (B,K,d)
        hMh = torch.sum(hk * v, dim=2)          # (B,K)
        log_den_terms = logpi - 0.5 * logdet_S - 0.5 * logdet_M + 0.5 * hMh - 0.5 * qk  # (B,K)
        # Z(t) is independent of x; we can take any batch element. Use first element for stability.
        logZ = torch.logsumexp(log_den_terms[0], dim=0)  # scalar

        log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
        return -0.5 * d * log2pi + log_num - logZ

    def p_star(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.exp(self.log_p_star(t, x))

    def xbar_star(self, t: Tensor) -> Tensor:
        """Expected state \bar x^{(*)}(t) (Eq. B.15)."""
        # Compute weights w_k(t) proportional to pi_k |Sk|^{-1/2} |Mk|^{-1/2} exp(1/2 h^T M^{-1} h - 1/2 q)
        # and then xbar = sum w_k M^{-1}h / sum w_k.
        # We implement for scalar t (or 1-element t); output is (d,).
        t = torch.as_tensor(t, dtype=self.coeffs.breaks.dtype, device=self.coeffs.breaks.device)
        if t.ndim == 0:
            tt = t.reshape(1)
        elif t.ndim == 1 and t.numel() == 1:
            tt = t
        else:
            raise ValueError("xbar_star currently expects scalar t (or shape (1,)).")

        B = 1
        d = self.target.d
        Mk, hk, Sk, qk = self._Mk_hk(tt)  # (1,K,*,*)
        logdet_S = torch.logdet(Sk)[0]    # (K,)
        logdet_M = torch.logdet(Mk)[0]    # (K,)
        logpi = torch.log(self.target.weights.to(dtype=logdet_S.dtype, device=logdet_S.device))  # (K,)

        v = torch.linalg.solve(Mk[0], hk[0].unsqueeze(-1)).squeeze(-1)  # (K,d)
        hMh = torch.sum(hk[0] * v, dim=1)  # (K,)

        logw = logpi - 0.5 * logdet_S - 0.5 * logdet_M + 0.5 * hMh - 0.5 * qk[0]  # (K,)
        w = torch.softmax(logw, dim=0)  # normalized
        return torch.sum(w.unsqueeze(-1) * v, dim=0)  # (d,)
