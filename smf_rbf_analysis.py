import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor

import time
import torch

def _maybe_sync(device):
    # Needed for accurate timing on GPU due to async kernels.
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _as_2d(x: Tensor) -> Tensor:
    if x.ndim == 1:
        return x[:, None]
    return x


def quantiles_sorted_vector(x_sorted: Tensor, n_or_q: Union[int, Sequence[float]]) -> Tensor:
    """
    Replacement for MATLAB quantiles_sorted_vector used by your code.

    x_sorted MUST be sorted ascending. Shape (N,) or (N,1).
    n_or_q:
      - int n: internal quantiles at i/(n+1), i=1..n
      - list/tuple: explicit quantiles in [0,1]
    """
    x_sorted = x_sorted.flatten()
    n = x_sorted.numel()

    if isinstance(n_or_q, int):
        if n_or_q <= 0:
            return x_sorted.new_zeros((0,))
        q = torch.arange(1, n_or_q + 1, device=x_sorted.device, dtype=torch.float32) / (n_or_q + 1.0)
    else:
        q = torch.tensor(list(n_or_q), device=x_sorted.device, dtype=torch.float32)

    idx = q * (n - 1)
    lo = torch.floor(idx).to(torch.long)
    hi = torch.clamp(lo + 1, max=n - 1)
    w = (idx - lo.to(idx.dtype))
    return (1.0 - w) * x_sorted[lo] + w * x_sorted[hi]


# def projected_newton_nonneg(
#     x0: Tensor,
#     obj: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
#     max_iter: int = 50,
#     tol: float = 1e-8,
#     ls_max_iter: int = 25,
#     ls_c: float = 1e-4,
#     ls_tau: float = 0.5,
#     # new:
#     damp0: float = 1e-10,
#     damp_max: float = 1e8,
#     symmetrize_H: bool = True,
# ) -> Tensor:
#     """
#     Projected Newton on nonnegative orthant with robust damped solve.
#     obj(x) -> (f, g, H)
#     """
#     x = torch.clamp(x0.clone(), min=0.0)

#     for _ in range(max_iter):
#         f, g, H = obj(x)

#         # projected gradient
#         pg = torch.where((x <= 0) & (g > 0), torch.zeros_like(g), g)
#         if torch.linalg.norm(pg).item() < tol:
#             break

#         if symmetrize_H:
#             H = 0.5 * (H + H.transpose(-1, -2))

#         I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

#         # --- robust damped Newton direction ---
#         # Try increasing diagonal shift until solve succeeds.
#         damp = damp0
#         p = None
#         while True:
#             try:
#                 p = torch.linalg.solve(H + damp * I, -g)
#                 break
#             except RuntimeError:
#                 damp *= 10.0
#                 if damp > damp_max:
#                     # Last resort: least-squares direction
#                     p = torch.linalg.lstsq(H + damp0 * I, -g).solution
#                     break

#         # line search
#         alpha = 1.0
#         gTp = (g * p).sum()
#         for _ls in range(ls_max_iter):
#             x_new = torch.clamp(x + alpha * p, min=0.0)
#             f_new, _, _ = obj(x_new)
#             if f_new <= f + ls_c * alpha * gTp:
#                 x = x_new
#                 break
#             alpha *= ls_tau
#         else:
#             x = torch.clamp(x + alpha * p, min=0.0)

#     return x

def projected_newton_nonneg(
    x0: Tensor,
    obj_fgh: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
    obj_f: Callable[[Tensor], Tensor],
    max_iter: int = 50,
    tol: float = 1e-8,
    ls_max_iter: int = 25,
    ls_c: float = 1e-4,
    ls_tau: float = 0.5,
    ridge: float = 1e-8,
) -> Tensor:
    """
    Projected Newton in nonnegative orthant.
    Key speed fix: line search evaluates ONLY f(x_new), not (f,g,H).
    """
    x = torch.clamp(x0.clone(), min=0.0)

    I = None  # allocate lazily once

    for _ in range(max_iter):
        f, g, H = obj_fgh(x)

        # projected gradient norm
        pg = torch.where((x <= 0) & (g > 0), torch.zeros_like(g), g)
        if torch.linalg.norm(pg).item() < tol:
            break

        if I is None or I.shape[0] != H.shape[0] or I.device != H.device or I.dtype != H.dtype:
            I = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

        # Newton direction with ridge that grows if needed
        p = None
        r = ridge
        for _try in range(6):
            try:
                p = torch.linalg.solve(H + r * I, -g)
                break
            except RuntimeError:
                r *= 10.0
        if p is None:
            # fallback: least-squares direction
            p = torch.linalg.lstsq(H + r * I, -g).solution

        # Armijo backtracking using f-only
        alpha = 1.0
        gTp = (g * p).sum()

        for _ls in range(ls_max_iter):
            x_new = torch.clamp(x + alpha * p, min=0.0)
            f_new = obj_f(x_new)
            if f_new <= f + ls_c * alpha * gTp:
                x = x_new
                break
            alpha *= ls_tau
        else:
            x = torch.clamp(x + alpha * p, min=0.0)

    return x




class NonMonotonePart:
    """
    Faithful port of NonMonotonePart.m
    """

    def __init__(self, order: Sequence[int], scalingRbf: float = 2.0):
        self.order = list(order)
        self.scalingRbf = float(scalingRbf)

        self.nvar = len(self.order)
        self.activeVars = [i for i, o in enumerate(self.order) if o > 0]
        self.ncoeff = int(sum(max(o, 0) for o in self.order))

        self.constTerm = torch.tensor(0.0)
        self.coeffs: List[Optional[Tensor]] = [None] * self.nvar
        self.centers: List[Optional[Tensor]] = [None] * self.nvar
        self.widths: List[Optional[Tensor]] = [None] * self.nvar

    def reset_coeffs(self, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        self.constTerm = torch.zeros((), device=device, dtype=dtype)
        for i in range(self.nvar):
            o = self.order[i]
            self.coeffs[i] = torch.zeros((o,), device=device, dtype=dtype) if o > 0 else None

    def centers_and_widths(self, X: Tensor, ncoeff: int) -> Tuple[Tensor, Tensor]:
        """
        MATLAB-faithful centers_and_widths:
        - sort X
        - centers from quantiles_sorted_vector(X_sorted, ncoeff)
        - widths per MATLAB rules
        - widths scaled by scalingRbf
        Returns:
          centers: (P,)
          widths:  (P,)
        """
        x = X.flatten().to(torch.float32)
        if x.numel() == 0:
            c = x.new_zeros((0,))
            w = x.new_zeros((0,))
            return c, w

        if ncoeff < 1:
            raise ValueError("ncoeff must be >= 1")

        x_sorted = torch.sort(x)[0]

        if ncoeff == 1:
            # qq = quantiles_sorted_vector(X,[.25, .5, .75]);
            qq = quantiles_sorted_vector(x_sorted, [0.25, 0.50, 0.75]).to(torch.float32)
            centers = qq[1:2]  # (1,)
            widths = ((qq[2] - qq[0]) / 2.0).reshape(1)
        else:
            # centers = quantiles_sorted_vector(X, ncoeff);
            centers = quantiles_sorted_vector(x_sorted, ncoeff).to(torch.float32)  # (P,)
            P = centers.numel()
            if P == 2:
                w0 = (centers[1] - centers[0]).abs()
                widths = w0.expand(2).clone()
            else:
                widths = centers.new_zeros((P,))
                widths[1:-1] = (centers[2:] - centers[:-2]) / 2.0
                widths[0] = centers[1] - centers[0]
                widths[-1] = centers[-1] - centers[-2]

        # scale the widths by scalingRbf
        widths = self.scalingRbf * widths

        # MATLAB implicitly assumes widths > 0; protect against exact duplicates
        # (this is the *minimal* safety guard; it preserves the MATLAB formula)
        widths = widths.clamp_min(1e-12)

        return centers, widths

    def _eval_rbf(self, X: Tensor, centers: Tensor, widths: Tensor) -> Tensor:
        X = X[:, None].to(torch.float32)
        c = centers[None, :].to(torch.float32)
        w = widths[None, :].to(torch.float32).clamp_min(1e-12)
        return torch.exp(-0.5 * ((X - c) / w) ** 2) / (w * math.sqrt(2.0 * math.pi))



    def setup(self, X: Tensor):
        X = _as_2d(X)

        # init coeffs ONCE (or if device changes) — warm-start thereafter
        if (self.constTerm.device != X.device) or (self.coeffs[0] is None):
            self.reset_coeffs(device=X.device, dtype=torch.float32)

        # centers/widths depend on X; you can recompute them each time OR cache them
        for i in self.activeVars:
            o = self.order[i]
            if o >= 2:
                c, w = self.centers_and_widths(X[:, i], o - 1)
                self.centers[i] = c.to(device=X.device, dtype=torch.float32)
                self.widths[i]  = w.to(device=X.device, dtype=torch.float32)


    # def _eval_rbf(self, X: Tensor, centers: Tensor, widths: Tensor) -> Tensor:
    #     X = X[:, None]
    #     c = centers[None, :]
    #     w = widths[None, :]
    #     return torch.exp(-0.5 * ((X - c) / w) ** 2)

    def basis_eval(self, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        cols = []
        for i in self.activeVars:
            o = self.order[i]
            xi = X[:, i]
            cols.append(xi[:, None])  # linear term
            if o >= 2:
                cols.append(self._eval_rbf(xi, self.centers[i], self.widths[i]))
        if not cols:
            return X.new_zeros((X.shape[0], 0), dtype=torch.float32)
        return torch.cat(cols, dim=1)

    def invalidate_cache(self):
        for i in self.activeVars:
            self.centers[i] = None
            self.widths[i] = None

    def eval(self, X: Tensor) -> Tensor:
        X = _as_2d(X)
        Psi = self.basis_eval(X)
        g_list = []
        for i in self.activeVars:
            g_list.append(self.coeffs[i].to(torch.float32))
        if not g_list:
            return self.constTerm.expand(X.shape[0]).to(torch.float32)
        g = torch.cat(g_list, dim=0)
        return (Psi @ g + self.constTerm).to(torch.float32)


class MonotonePart:
    """
    Faithful port of MonotonePart.m
    """

    def __init__(
        self,
        order: int,
        scalingRbf: float = 2.0,
        approx_inverse_npts: int = 250,
    ):
        self.order = int(order)
        self.scalingRbf = float(scalingRbf)
        self.approx_inverse_npts = int(approx_inverse_npts)
        self.kappa = 5.0
        self.npoints_interp = 250

        self.ncoeff = 1 if self.order == 1 else (self.order + 1)
        self.coeffs: Optional[Tensor] = None
        self.centers: Optional[Tensor] = None
        self.widths: Optional[Tensor] = None
        self.xx: Optional[Tensor] = None
        self.yy: Optional[Tensor] = None
    
    def invalidate_cache(self):
        self.centers = None
        self.widths = None
        self.xx = None
        self.yy = None

    def reset(self, device=None, dtype=None, clear_cache: bool = False):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        self.coeffs = torch.zeros((self.ncoeff,), device=device, dtype=dtype)
        if clear_cache:
            self.centers = None
            self.widths = None
            self.xx = None
            self.yy = None

    def centers_and_widths(self, X: Tensor, ncoeff: Union[int, Sequence[float]]) -> Tuple[Tensor, Tensor]:
        """
        Robust centers/widths:
        - centers from quantiles
        - if quantiles duplicate, fall back to evenly spaced centers over [min,max]
        - widths from mean center spacing, with a lower bound based on data scale
        """
        x = X.flatten().to(torch.float32)
        if x.numel() == 0:
            c = x.new_zeros((0,))
            w = x.new_ones((0,))
            return c, w

        # --- centers ---
        if isinstance(ncoeff, int):
            P = int(ncoeff)
            if P <= 0:
                c = x.new_zeros((0,))
            else:
                q = torch.arange(1, P + 1, device=x.device, dtype=torch.float32) / (P + 1.0)
                c = torch.quantile(x, q, interpolation="linear")
        else:
            q = torch.tensor(list(ncoeff), device=x.device, dtype=torch.float32)
            c = torch.quantile(x, q, interpolation="linear")
            P = c.numel()

        if P == 0:
            return c, x.new_zeros((0,))

        # If quantile centers are not strictly increasing, replace with linspace centers
        x_min = x.min()
        x_max = x.max()
        rng = (x_max - x_min).abs()

        # detect duplicates / non-increasing centers
        if P > 1:
            dc = c[1:] - c[:-1]
            if not torch.all(dc > 0):
                if rng.item() == 0.0:
                    # constant variable: all centers identical
                    c = x_min.expand_as(c).clone()
                else:
                    # strictly increasing centers inside (min,max)
                    c = torch.linspace(x_min, x_max, P + 2, device=x.device, dtype=torch.float32)[1:-1]

        # --- widths ---
        if P == 1:
            base = x.new_tensor(1.0)
        else:
            dc = c[1:] - c[:-1]
            base = dc.mean()

        # lower bound based on scale of x
        # (these are small but prevent w=0 and therefore 0/0)
        std = x.std(unbiased=False)
        eps_scale = torch.maximum(
            x.new_tensor(1e-12),
            torch.maximum(1e-6 * rng, 1e-3 * std),
        )

        w = (self.scalingRbf * base).expand_as(c).clone()
        w = torch.clamp(w, min=float(eps_scale.item()))
        return c, w



    def _eval_monotone_rbfs(self, X: Tensor, centers: Tensor, widths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        MATLAB-faithful version of:
        eval_monotone_rbfs
        grad_x_monotone_rbfs
        hess_x_monotone_rbfs

        centers, widths are 1D tensors of length P = self.ncoeff (>=3 when order>1)
        X is (N,) or (N,1)
        """
        X = X.reshape(-1, 1).to(torch.float32)              # (N,1)
        c = centers.reshape(1, -1).to(torch.float32)        # (1,P)
        w = widths.reshape(1, -1).to(torch.float32)         # (1,P)

        # guard widths (MATLAB assumes positive)
        w = w.clamp_min(1e-12)

        delta = (X - c) / (w * math.sqrt(2.0))              # (N,P)
        erf_d = torch.erf(delta)
        exp_d = torch.exp(-(delta ** 2))

        N, P = delta.shape
        if P < 3:
            raise ValueError(f"MATLAB MonotonePart requires >=3 centers/widths when order>1; got P={P}")

        sqrt2 = math.sqrt(2.0)
        sqrt2_over_pi = math.sqrt(2.0 / math.pi)
        inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)

        f = X.new_zeros((N, P), dtype=torch.float32)
        df = X.new_zeros((N, P), dtype=torch.float32)
        d2f = X.new_zeros((N, P), dtype=torch.float32)

        # ---- f (MATLAB eval_monotone_rbfs) ----
        # f(:,1)
        f[:, 0] = 0.5 * (
            sqrt2 * w[:, 0] * delta[:, 0] * (1.0 - erf_d[:, 0])
            - w[:, 0] * sqrt2_over_pi * exp_d[:, 0]
        )
        # f(:,2:end-1)
        f[:, 1:-1] = 0.5 * (1.0 + erf_d[:, 1:-1])
        # f(:,end)
        f[:, -1] = 0.5 * (
            sqrt2 * w[:, -1] * delta[:, -1] * (1.0 + erf_d[:, -1])
            + w[:, -1] * sqrt2_over_pi * exp_d[:, -1]
        )

        # ---- df (MATLAB grad_x_monotone_rbfs) ----
        df[:, 0] = 0.5 * (1.0 - erf_d[:, 0])
        df[:, 1:-1] = exp_d[:, 1:-1] / (w[:, 1:-1] * math.sqrt(2.0 * math.pi))
        df[:, -1] = 0.5 * (1.0 + erf_d[:, -1])

        # ---- d2f (MATLAB hess_x_monotone_rbfs) ----
        d2f[:, 0] = -exp_d[:, 0] / (w[:, 0] * math.sqrt(2.0 * math.pi))
        df_mid = exp_d[:, 1:-1] / (w[:, 1:-1] * math.sqrt(2.0 * math.pi))
        d2f[:, 1:-1] = -df_mid * delta[:, 1:-1] * math.sqrt(2.0) / w[:, 1:-1]
        d2f[:, -1] = exp_d[:, -1] / (w[:, -1] * math.sqrt(2.0 * math.pi))

        return f, df, d2f


    def basis_eval(self, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        x = X[:, 0]
        if self.order == 1:
            return x[:, None]
        if self.centers is None or self.widths is None:
            c, w = self.centers_and_widths(x, self.ncoeff)
            self.centers = c.to(device=x.device)
            self.widths = w.to(device=x.device)
        f, _, _ = self._eval_monotone_rbfs(x, self.centers, self.widths)
        return f

    def basis_grad_x(self, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        x = X[:, 0]
        if self.order == 1:
            return torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float32)
        if self.centers is None or self.widths is None:
            c, w = self.centers_and_widths(x, self.ncoeff)
            self.centers = c.to(device=x.device)
            self.widths = w.to(device=x.device)
        _, df, _ = self._eval_monotone_rbfs(x, self.centers, self.widths)
        return df

    def basis_hess_x(self, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        x = X[:, 0]
        if self.order == 1:
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.float32)
        if self.centers is None or self.widths is None:
            c, w = self.centers_and_widths(x, self.ncoeff)
            self.centers = c.to(device=x.device)
            self.widths = w.to(device=x.device)
        _, _, d2f = self._eval_monotone_rbfs(x, self.centers, self.widths)
        return d2f

    @torch.no_grad()
    def inverse(self, y: Tensor, X: Tensor, coeffs: Optional[Tensor] = None) -> Tensor:
        y = y.flatten().to(torch.float32)
        X = _as_2d(X).to(torch.float32)
        x_samples = X[:, 0]
        device = x_samples.device

        coeffs = coeffs if coeffs is not None else self.coeffs
        assert coeffs is not None

        # Build (xx, yy) lookup table if needed
        rebuild = (self.xx is None) or (self.yy is None) or (self.xx.device != device)
        if rebuild:
            # MATLAB domain:
            lbound = self.centers[0] - self.kappa * self.widths[0]
            ubound = self.centers[-1] + self.kappa * self.widths[-1]
            xx = torch.linspace(lbound, ubound, self.npoints_interp, device=device, dtype=torch.float32)
            yy = self.basis_eval(xx[:, None]) @ coeffs.to(torch.float32)
            self.xx, self.yy = xx, yy

        xx, yy = self.xx, self.yy

        # Enforce nondecreasing yy in xx-order (fix tiny numerical violations)
        yy_mono = torch.cummax(yy, dim=0).values

        # If yy is (almost) flat, inversion is ill-posed -> do NOT collapse the ensemble
        y_range = (yy_mono[-1] - yy_mono[0]).abs()
        if y_range < 1e-8:
            # safest fallback: identity-ish (return original samples)
            # If you prefer: return x_samples.mean().expand_as(y) for a deterministic scalar
            return x_samples.clone()

        # Make it strictly increasing to keep searchsorted stable on plateaus
        eps = 1e-12 * torch.arange(yy_mono.numel(), device=device, dtype=torch.float32)
        yy_mono = yy_mono + eps

        
        # Clamp and invert via linear interpolation
        y_clamped = torch.clamp(y, min=yy_mono[0].item(), max=yy_mono[-1].item())
        idx = torch.searchsorted(yy_mono, y_clamped, right=False)
        idx = torch.clamp(idx, 1, yy_mono.numel() - 1)

        y0, y1 = yy_mono[idx - 1], yy_mono[idx]
        x0, x1 = xx[idx - 1], xx[idx]
        t = (y_clamped - y0) / (y1 - y0 + 1e-12)
        return x0 + t * (x1 - x0)



class TransportMapComponent:
    """
    Faithful port of TransportMapComponent.m
    """

    def __init__(self, order: Sequence[int], lambda_: float = 0.0, delta: float = 1e-8, scalingRbf: float = 2.0):
        self.order = list(order)
        self.d = len(order)
        self.lambda_ = float(lambda_)
        self.delta = float(delta)
        self.scalingRbf = float(scalingRbf)

        self.OffD_nvar = self.d - 1
        self.Diag_nvar = 1

        self.OffD = NonMonotonePart(self.order[: self.OffD_nvar], scalingRbf=self.scalingRbf) if self.OffD_nvar > 0 else None
        self.Diag = MonotonePart(self.order[-1], scalingRbf=self.scalingRbf)
        self.Diag.reset()

    def invalidate_cache(self):
        # invalidate cache for off-diagonal basis (depends on current X)
        if self.OffD is not None:
            self.OffD.invalidate_cache()
        # invalidate cache for diagonal basis + inverse table (depends on current x_k)
        self.Diag.invalidate_cache()

    @torch.no_grad()
    def eval(self, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        xk = X[:, -1:]
        Psi_mon = self.Diag.basis_eval(xk)
        y = Psi_mon @ self.Diag.coeffs.to(X.device, dtype=torch.float32)
        if self.OffD_nvar > 0:
            y = y + self.OffD.eval(X[:, :-1])
        return y
    @torch.no_grad()
    def optimize(self, X: Tensor):
        """
        MATLAB-faithful TransportMapComponent.optimize.

        Matches:
        - construct_basis() behavior (centers/widths recomputed from current X)
        - Psi_mon / dPsi_mon normalization with population std
        - QR projection using [Psi_offD; sqrt(lambda) I]
        - A formation from projected residual Asqrt
        - setup_and_run_optim objective structure
        - g_off = -R\(Q1' * Psi_mon * g_mon)
        - rescaling + constTerm exactly as MATLAB
        """
        X = _as_2d(X)
        device = X.device
        X = X.to(torch.float32)
        N = X.shape[0]

        if X.shape[1] != self.d:
            raise ValueError(f"Dimension mismatch: got X.shape[1]={X.shape[1]} but component d={self.d}")

        # -------------------------
        # Build diagonal (monotone) basis from CURRENT x_k  (MATLAB construct_basis)
        # -------------------------
        xk = X[:, -1:]  # (N,1)

        if self.Diag.coeffs is None or self.Diag.coeffs.numel() != self.Diag.ncoeff or self.Diag.coeffs.device != device:
            self.Diag.reset(device=device, dtype=torch.float32, clear_cache=True)

        if self.Diag.order > 1:
            c, w = self.Diag.centers_and_widths(xk[:, 0], self.Diag.ncoeff)
            # MATLAB expects row vectors; we store 1D tensors
            self.Diag.centers = c.to(device=device, dtype=torch.float32)
            self.Diag.widths  = w.to(device=device, dtype=torch.float32)
            # also invalidate inverse table since centers/widths changed
            self.Diag.xx = None
            self.Diag.yy = None

        Psi_mon  = self.Diag.basis_eval(xk)       # (N, p)
        dPsi_mon = self.Diag.basis_grad_x(xk)     # (N, p)
        p = Psi_mon.shape[1]

        # Normalize monotone basis (MATLAB: mean=sum/N, std=sqrt(sum((.)^2)/N))
        meanPsi = Psi_mon.mean(dim=0, keepdim=True)
        stdPsi  = ( (Psi_mon - meanPsi).pow(2).mean(dim=0, keepdim=True) ).sqrt()
        # MATLAB can divide by zero if a basis column is constant; prevent NaNs with minimal deviation
        stdPsi  = stdPsi.clamp_min(1e-12)

        Psi_mon_std  = (Psi_mon - meanPsi) / stdPsi
        dPsi_mon_std = dPsi_mon / stdPsi

        # -------------------------
        # Off-diagonal basis + QR projection (MATLAB exact)
        # -------------------------
        if self.OffD_nvar != 0:
            assert self.OffD is not None
            self.OffD.setup(X[:, :-1])  # recompute centers/widths from CURRENT offdiag samples
            Psi_offD = self.OffD.basis_eval(X[:, :-1])  # (N, q)
            q = Psi_offD.shape[1]

            meanX = Psi_offD.mean(dim=0, keepdim=True)
            stdX  = ( (Psi_offD - meanX).pow(2).mean(dim=0, keepdim=True) ).sqrt().clamp_min(1e-12)
            Psi_offD_std = (Psi_offD - meanX) / stdX

            # QR on [Psi_offD_std; sqrt(lambda)*I]  (MATLAB: qr(...,0))
            if q > 0:
                Aqr = torch.cat(
                    [Psi_offD_std, math.sqrt(self.lambda_) * torch.eye(q, device=device, dtype=torch.float32)],
                    dim=0
                )  # (N+q, q)

                Q, R = torch.linalg.qr(Aqr, mode="reduced")  # Q:(N+q,q), R:(q,q)
                Q1 = Q[:N, :]                                # first N rows

                Asqrt = Psi_mon_std - Q1 @ (Q1.T @ Psi_mon_std)
                A = (Asqrt.T @ Asqrt) / float(N)
            else:
                # degenerate offdiag
                meanX = Psi_mon_std.new_zeros((1, 0))
                stdX  = Psi_mon_std.new_ones((1, 0))
                R = None
                Q1 = None
                A = (Psi_mon_std.T @ Psi_mon_std) / float(N)
        else:
            meanX = None
            stdX  = None
            q = 0
            R = None
            Q1 = None
            A = (Psi_mon_std.T @ Psi_mon_std) / float(N)

        # -------------------------
        # Solve diagonal coefficients (MATLAB)
        # -------------------------
        if self.Diag.order == 1:
            # A should be scalar
            a00 = A.reshape(-1)[0].clamp_min(1e-18)
            g_mon = torch.sqrt(1.0 / a00).reshape(1)  # (1,)
        else:
            # MATLAB setup_and_run_optim:
            # A = A + (lambda/N)*I ; b = delta*sum(A,2) ; projectedNewton ; xopt = xopt + delta
            I = torch.eye(p, device=device, dtype=torch.float32)
            A_reg = A + (self.lambda_ / float(N)) * I
            b = self.delta * A_reg.sum(dim=1)

            dPsi_sum = dPsi_mon_std.sum(dim=1)

            def obj_fgh(xvec: Tensor):
                Ax = A_reg @ xvec
                dS = (dPsi_mon_std @ xvec) + self.delta * dPsi_sum
                dS = dS.clamp_min(1e-20)

                fx = 0.5 * (xvec @ Ax) - torch.sum(torch.log(dS)) / float(N) + (xvec @ b)

                dPsi_dS = dPsi_mon_std / dS[:, None]
                gx = Ax - dPsi_dS.sum(dim=0) / float(N) + b

                Hx = A_reg + (dPsi_dS.T @ dPsi_dS) / float(N)
                Hx = 0.5 * (Hx + Hx.T)  # tiny symmetrization for numeric stability
                return fx, gx, Hx

            def obj_f(xvec: Tensor):
                Ax = A_reg @ xvec
                dS = (dPsi_mon_std @ xvec) + self.delta * dPsi_sum
                dS = dS.clamp_min(1e-20)
                fx = 0.5 * (xvec @ Ax) - torch.sum(torch.log(dS)) / float(N) + (xvec @ b)
                return fx

            if (self.Diag.coeffs is not None
                and self.Diag.coeffs.numel() == p
                and torch.isfinite(self.Diag.coeffs).all()):
                g_prev_std = (self.Diag.coeffs * stdPsi.flatten()).clamp_min(self.delta)
                x0 = torch.clamp(g_prev_std - self.delta, min=0.0)
            else:
                x0 = torch.ones((p,), device=device, dtype=torch.float32)
            xopt = projected_newton_nonneg(
                x0, obj_fgh, obj_f,
                max_iter=15, tol=1e-6,
                ls_max_iter=25, ls_c=1e-4, ls_tau=0.5,
                ridge=1e-12
            )
            g_mon = xopt + self.delta  # MATLAB: xopt = xopt + delta

        # -------------------------
        # Solve off-diagonal coefficients (MATLAB: g_off = -R\(Q1' * Psi_mon * g_mon))
        # -------------------------
        if self.OffD_nvar != 0 and q > 0:
            # v = Q1' * (Psi_mon_std * g_mon)
            v = Q1.T @ (Psi_mon_std @ g_mon)  # (q,)
            # Solve R g_off = v  (R is upper triangular)
            g_off = -torch.linalg.solve_triangular(R, v[:, None], upper=True).squeeze(1)  # (q,)

            # rescale to original coordinates
            g_mon = g_mon / stdPsi.flatten()
            g_off = g_off / stdX.flatten()

            constTerm = -(meanPsi.flatten() @ g_mon) - (meanX.flatten() @ g_off)

        else:
            # only diagonal
            g_mon = g_mon / stdPsi.flatten()
            g_off = None
            constTerm = -(meanPsi.flatten() @ g_mon)

        # -------------------------
        # Write back coefficients exactly like MATLAB assigns
        # -------------------------
        self.Diag.coeffs = g_mon.to(device=device, dtype=torch.float32)

        if self.OffD_nvar != 0:
            if self.OffD.coeffs[0] is None or self.OffD.constTerm.device != device:
                self.OffD.reset_coeffs(device=device, dtype=torch.float32)

            if g_off is not None:
                counter = 0
                for i in self.OffD.activeVars:
                    o = self.OffD.order[i]
                    self.OffD.coeffs[i][:] = g_off[counter:counter + o]
                    counter += o

            self.OffD.constTerm = constTerm.to(device=device, dtype=torch.float32)


    @torch.no_grad()
    def inverse(self, y: Tensor, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        y = y.flatten().to(torch.float32)
        if self.OffD_nvar > 0:
            y_tilde = y - self.OffD.eval(X[:, :-1])
        else:
            y_tilde = y
        xk = self.Diag.inverse(y_tilde, X[:, -1:], coeffs=self.Diag.coeffs)
        Xinv = X.clone()
        Xinv[:, -1] = xk
        return Xinv


class TransportMap:
    """
    Faithful port of TransportMap.m
    """

    def __init__(self, order, lambda_=0.0, delta=1e-8, scalingRbf=2.0):
        self.order = [list(o) for o in order]
        self.d = len(self.order)

        # Enforce MATLAB triangular structure: len(order[k]) == k+1
        for k in range(self.d):
            if len(self.order[k]) != (k + 1):
                raise ValueError(f"order[{k}] must have length {k+1}, got {len(self.order[k])}")

        self.Components = [
            TransportMapComponent(self.order[k], lambda_=lambda_, delta=delta, scalingRbf=scalingRbf)
            for k in range(self.d)
        ]

    @torch.no_grad()
    def optimize(self, X: Tensor, nonIdComp: Optional[Sequence[int]] = None):
        X = _as_2d(X).to(torch.float32)
        nonIdComp = list(nonIdComp) if nonIdComp is not None else list(range(1, self.d + 1))
        for k1 in nonIdComp:  # MATLAB-style 1-based
            self.Components[k1 - 1].optimize(X[:, :k1])

    @torch.no_grad()
    def eval_map(self, X: Tensor) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        Z = X.new_zeros((X.shape[0], self.d), dtype=torch.float32)
        for k in range(self.d):
            Z[:, k] = self.Components[k].eval(X[:, :k + 1]).flatten()
        return Z

    @torch.no_grad()
    def inverse_map(self, Z: Tensor, X0: Optional[Tensor] = None) -> Tensor:
        Z = _as_2d(Z).to(torch.float32)

        if X0 is None:
            # MATLAB-faithful: start from zeros
            X = Z.new_zeros((Z.shape[0], self.d), dtype=torch.float32)
        else:
            X = _as_2d(X0).to(torch.float32).clone()

        for k in range(self.d):
            X[:, :k + 1] = self.Components[k].inverse(Z[:, k], X[:, :k + 1])
        return X

    def invalidate_cache(self):
        for comp in self.Components:
            comp.invalidate_cache()



class StochasticMapFilterPy:
    """
    Port of the high-level logic in StochasticMapFilter.m (scalar-sequential assimilation).
    """

    def __init__(
        self,
        distMat: Optional[Tensor],
        offdiag_rad: float,
        order_all: int,
        diag_order: int,
        nonId_radius: int,
        lambda_: float = 0.0,
        delta: float = 1e-8,
        scalingRbf: float = 2.0,
        M: Optional[int] = None,
    ):
        self.distMat = distMat
        self.offdiag_rad = float(offdiag_rad)
        self.order_all = int(order_all)
        self.diag_order = int(diag_order)
        self.nonId_radius = int(nonId_radius)
        self.lambda_ = float(lambda_)
        self.delta = float(delta)
        self.scalingRbf = float(scalingRbf)
        self.M = M
        self.TM: Optional[TransportMap] = None

    def _generate_realizations(self, N: int, M: int, device) -> Tensor:
        if M >= N:
            return torch.arange(N, device=device)

        # if you really need without-replacement and M close to N:
        if M > 0.5 * N:
            return torch.randperm(N, device=device)[:M]

        # otherwise: with-replacement is fine and massively faster
        return torch.randint(0, N, (M,), device=device)


    def _apply_H(self, X: Tensor, H: Union[Tensor, Callable[[Tensor], Tensor]]) -> Tensor:
        if callable(H):
            Y = H(X)
            if isinstance(Y, (tuple, list)):
                Y = Y[0]
            if Y.dim() == 3 and Y.shape[0] == 1:
                Y = Y.squeeze(0)
            elif Y.dim() == 3 and Y.shape[1] == 1:
                Y = Y.squeeze(1)
            return _as_2d(Y).to(torch.float32)

        Ht = torch.as_tensor(H, device=X.device, dtype=torch.float32)
        if Ht.ndim != 2:
            raise ValueError("H must be 2D matrix or callable")
        if Ht.shape[1] == X.shape[1]:
            return X.to(torch.float32) @ Ht.T
        if Ht.shape[0] == X.shape[1]:
            return X.to(torch.float32) @ Ht
        raise ValueError(f"H has incompatible shape {Ht.shape} for state dim {X.shape[1]}")

    def _build_order(self, K: int, distMatK: Optional[Tensor]) -> List[List[int]]:
        # component 1: y only
        order: List[List[int]] = [[self.diag_order]]

        if distMatK is None:
            for i in range(1, K + 1):
                deps = [self.order_all] + [self.order_all] * (i - 1) + [self.diag_order]
                order.append(deps)
            return order

        for i in range(1, K + 1):
            deps = [self.order_all]
            if i > 1:
                drow = distMatK[i - 1, : i - 1]
                deps += [self.order_all if float(drow[j].item()) <= self.offdiag_rad else 0 for j in range(i - 1)]
            deps.append(self.diag_order)
            order.append(deps)
        return order

    def sample_posterior(
        self,
        X: Tensor,                          # (N,d)
        y: Tensor,                          # (m,)
        H: Union[Tensor, Callable[[Tensor], Tensor]],
        sigma_y: Union[float, Tensor],
        perm: Optional[Tensor] = None,
    ) -> Tensor:
        X = _as_2d(X).to(torch.float32)
        y = y.flatten().to(torch.float32)
        N, d = X.shape

        K = min(self.nonId_radius, d)
        perm = torch.arange(d, device=X.device) if perm is None else perm.to(torch.long)
        Xp = X[:, perm]
        Xsub = Xp[:, :K]

        Y_pred = self._apply_H(X, H)  # (N,m)
        m = Y_pred.shape[1]

        distMatK = None
        if self.distMat is not None:
            Dfull = self.distMat.to(device=X.device, dtype=torch.float32)
            Dperm = Dfull[perm][:, perm]
            distMatK = Dperm[:K, :K]
        
        order = self._build_order(K, distMatK)
        TM = TransportMap(order, lambda_=self.lambda_, delta=self.delta, scalingRbf=self.scalingRbf)
        
        for j in range(m):
            yj = y[j]
            sigma_j = float(sigma_y[j].item()) if isinstance(sigma_y, torch.Tensor) and sigma_y.numel() > 1 else float(sigma_y)

            M = self.M or N
            idx = self._generate_realizations(N, M, device=X.device)
            Yj_synth = Y_pred[idx, j] + float(sigma_j) * torch.randn((M,), device=X.device, dtype=torch.float32)
            mapInput = torch.cat([Yj_synth[:, None], Xsub[idx]], dim=1)  # (M,K+1)

            

            if j == 0:
                TM.invalidate_cache()
            
            TM.optimize(mapInput, nonIdComp=list(range(1, K + 2)))

            mapOutput = TM.eval_map(mapInput)
            mapOutput[:, 0] = yj
            mapInput_post = TM.inverse_map(mapOutput, mapInput)
            Xsub[idx] = mapInput_post[:, 1:]

            self.TM = TM

        Xp[:, :K] = Xsub
        invperm = torch.argsort(perm)
        return Xp[:, invperm].to(dtype=X.dtype)

    def sample_posterior(
        self,
        X: torch.Tensor,                          # (N,d)
        y: torch.Tensor,                          # (m,)
        H,                                        # Tensor or callable
        sigma_y,
        perm: Optional[Tensor] = None,
        *,
        timing: bool = True,
        timing_every_j: int = 1,                  # print every j (set to 5/10 to reduce spam)
    ) -> torch.Tensor:
        X = _as_2d(X).to(torch.float32)
        y = y.flatten().to(torch.float32)
        N, d = X.shape
        device = X.device

        def stamp():
            _maybe_sync(device)
            return time.perf_counter()

        t0_total = stamp()

        # -----------------------
        # Setup / permute
        # -----------------------
        t0 = stamp()
        K = min(self.nonId_radius, d)
        perm = torch.arange(d, device=device) if perm is None else perm.to(torch.long)
        Xp = X[:, perm]
        Xsub = Xp[:, :K]
        t_setup = stamp() - t0

        # -----------------------
        # Apply H once
        # -----------------------
        t0 = stamp()
        Y_pred = self._apply_H(X, H)  # (N,m)
        m = Y_pred.shape[1]
        t_applyH = stamp() - t0

        # -----------------------
        # distMatK once
        # -----------------------
        t0 = stamp()
        distMatK = None
        if self.distMat is not None:
            Dfull = self.distMat.to(device=device, dtype=torch.float32)
            Dperm = Dfull[perm][:, perm]
            distMatK = Dperm[:K, :K]
        t_dist = stamp() - t0

        # -----------------------
        # Build order + TM once
        # -----------------------
        t0 = stamp()
        order = self._build_order(K, distMatK)
        TM = TransportMap(order, lambda_=self.lambda_, delta=self.delta, scalingRbf=self.scalingRbf)
        t_tm_init = stamp() - t0

        # -----------------------
        # Main scalar loop
        # -----------------------
        t_loop_total = 0.0
        t_gen_total = 0.0
        t_synth_total = 0.0
        t_mapinput_total = 0.0
        t_opt_total = 0.0
        t_eval_total = 0.0
        t_inv_total = 0.0
        t_assign_total = 0.0
        t_cache_total = 0.0

        for j in range(m):
            tj0 = stamp()

            # --- generate idx ---
            t0 = stamp()
            M = self.M or N
            idx = self._generate_realizations(N, M, device=device)
            t_gen = stamp() - t0

            # --- synthetic obs ---
            t0 = stamp()
            yj = y[j]
            if isinstance(sigma_y, torch.Tensor) and sigma_y.numel() > 1:
                sigma_j = float(sigma_y[j].item())
            else:
                sigma_j = float(sigma_y)
            Yj_synth = Y_pred[idx, j] + sigma_j * torch.randn((M,), device=device, dtype=torch.float32)
            t_synth = stamp() - t0

            # --- mapInput build ---
            t0 = stamp()
            mapInput = torch.cat([Yj_synth[:, None], Xsub[idx]], dim=1)  # (M,K+1)
            t_mapinput = stamp() - t0

            # --- cache invalidation (if you insist on keeping it) ---
            t0 = stamp()
            # TM.invalidate_cache()   # <-- comment/uncomment to measure its cost
            t_cache = stamp() - t0

            # --- optimize ---
            t0 = stamp()
            TM.optimize(mapInput, nonIdComp=list(range(1, K + 2)))
            t_opt = stamp() - t0

            # --- eval map ---
            t0 = stamp()
            mapOutput = TM.eval_map(mapInput)
            t_eval = stamp() - t0

            # --- inverse map ---
            t0 = stamp()
            mapOutput[:, 0] = yj
            mapInput_post = TM.inverse_map(mapOutput, mapInput)
            t_inv = stamp() - t0

            # --- assign back ---
            t0 = stamp()
            Xsub[idx] = mapInput_post[:, 1:]
            t_assign = stamp() - t0

            tj = stamp() - tj0

            # accumulate totals
            t_loop_total += tj
            t_gen_total += t_gen
            t_synth_total += t_synth
            t_mapinput_total += t_mapinput
            t_cache_total += t_cache
            t_opt_total += t_opt
            t_eval_total += t_eval
            t_inv_total += t_inv
            t_assign_total += t_assign

            # if timing and (j % timing_every_j == 0 or j == m - 1):
            #     print(
            #         f"[SMF timing] j={j:3d}/{m-1:3d}  total={tj*1e3:8.2f} ms | "
            #         f"idx={t_gen*1e3:7.2f}  synth={t_synth*1e3:7.2f}  "
            #         f"mapIn={t_mapinput*1e3:7.2f}  cache={t_cache*1e3:7.2f}  "
            #         f"opt={t_opt*1e3:9.2f}  eval={t_eval*1e3:7.2f}  inv={t_inv*1e3:7.2f}  "
            #         f"assign={t_assign*1e3:7.2f}"
            #     )

        # -----------------------
        # Pack / unpermute
        # -----------------------
        t0 = stamp()
        Xp[:, :K] = Xsub
        invperm = torch.argsort(perm)
        out = Xp[:, invperm].to(dtype=X.dtype)
        t_pack = stamp() - t0

        t_total = stamp() - t0_total

        # if timing:
        #     print("\n========== SMF timing summary ==========")
        #     print(f"setup/perm:   {t_setup:8.4f} s")
        #     print(f"apply_H:      {t_applyH:8.4f} s")
        #     print(f"distMatK:     {t_dist:8.4f} s")
        #     print(f"TM init:      {t_tm_init:8.4f} s")
        #     print(f"loop total:   {t_loop_total:8.4f} s  (m={m}, K={K}, M={self.M or N})")
        #     print(f"  idx:        {t_gen_total:8.4f} s")
        #     print(f"  synth:      {t_synth_total:8.4f} s")
        #     print(f"  mapInput:   {t_mapinput_total:8.4f} s")
        #     print(f"  cache:      {t_cache_total:8.4f} s")
        #     print(f"  optimize:   {t_opt_total:8.4f} s")
        #     print(f"  eval_map:   {t_eval_total:8.4f} s")
        #     print(f"  inverse:    {t_inv_total:8.4f} s")
        #     print(f"  assign:     {t_assign_total:8.4f} s")
        #     print(f"pack/unperm:  {t_pack:8.4f} s")
        #     print(f"TOTAL:        {t_total:8.4f} s")
        #     print("========================================\n")

        return out



def smf_transport_update(
    xf: Tensor,                           # (N,d) or (B,N,d)
    y: Tensor,                            # (m,) or (B,m)
    H: Union[Tensor, Callable[[Tensor], Tensor]],
    sigma_y: Union[float, Tensor],
    *,
    distMat: Optional[Tensor],
    offdiag_rad: float,
    p_rbf: int,
    diag_order: int = 2,
    nonId_radius: Optional[int] = None,
    M: Optional[int] = None,
    lambda_: float = 0.0,
    delta: float = 1e-8,
    scalingRbf: float = 2.0,
) -> Tensor:
    """
    Batched wrapper (matches MATLAB scalar-sequential assimilation semantics).
    """
    nonId_radius = xf.shape[-1] if nonId_radius is None else int(nonId_radius)
    order_all = 1 + int(p_rbf)

    if xf.ndim == 2:
        smf = StochasticMapFilterPy(
            distMat=distMat,
            offdiag_rad=offdiag_rad,
            order_all=order_all,
            diag_order=diag_order,
            nonId_radius=nonId_radius,
            lambda_=lambda_,
            delta=delta,
            scalingRbf=scalingRbf,
            M=M,
        )
        return smf.sample_posterior(xf, y, H, sigma_y)

    if xf.ndim == 3:
        B = xf.shape[0]
        xa = xf.clone()
        for b in range(B):
            print(b)
            smf = StochasticMapFilterPy(
                distMat=distMat,
                offdiag_rad=offdiag_rad,
                order_all=order_all,
                diag_order=diag_order,
                nonId_radius=nonId_radius,
                lambda_=lambda_,
                delta=delta,
                scalingRbf=scalingRbf,
                M=M,
            )
            xa[b] = smf.sample_posterior(xf[b], y[b], H, sigma_y)
        return xa

    # raise ValueError("xf must be (N,d) or (B,N,d)")



