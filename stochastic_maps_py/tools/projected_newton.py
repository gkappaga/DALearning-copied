from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import torch


Objective = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
Method = Literal["true_hessian", "mod_hessian", "gradient"]


@dataclass
class ProjectedNewtonResult:
    x_opt: torch.Tensor
    message: str
    rel_delta_obj: float
    projected_grad_norm: float
    iterations: int


def projected_newton(
    x0: torch.Tensor,
    objective: Objective,
    *,
    rtol_obj: float = 1e-6,
    rtol_grad: float = 1e-6,
    max_iter: int = 30,
    method: Method = "true_hessian",
) -> ProjectedNewtonResult:
    """
    Translate the MATLAB ``projectedNewton`` solver for strictly convex problems.

    Parameters
    ----------
    x0:
        Feasible initial point (non-negative components).
    objective:
        Callable returning ``(value, gradient, hessian)`` at a given iterate.
    """
    if x0.ndim != 1:
        raise ValueError("`x0` must be a column vector.")
    if torch.any(x0 < 0):
        raise ValueError("Initial iterate must be feasible (>= 0).")

    xk = x0.clone().detach().double()
    Jk, gk, Hk = _evaluate_objective(objective, xk)
    norm_pg0 = _projected_gradient(xk, gk).norm()
    tol_grad = norm_pg0 * rtol_grad
    rel_delta = rtol_obj + 1.0
    Jprev = Jk
    iterations = 0

    while rel_delta > rtol_obj and _projected_gradient(xk, gk).norm() > tol_grad and iterations < max_iter:
        wk = torch.norm(xk - torch.maximum(torch.zeros_like(xk), xk - gk))
        epsk = min(0.01, wk.item())
        active = torch.where((xk <= epsk) & (gk > 0))[0]
        if active.numel() > 0:
            diag_entries = torch.diagonal(Hk)
            zk = torch.zeros_like(xk)
            zk[active] = diag_entries[active]
            Hk = Hk.clone()
            Hk[:, active] = 0.0
            Hk[active, :] = 0.0
            Hk += torch.diag(zk)

        pk = _compute_direction(method, Hk, gk)
        alpha = _armijo_line_search(xk, gk, pk, Jk, active, objective)
        xk = torch.maximum(torch.zeros_like(xk), xk - alpha * pk)
        Jk, gk, Hk = _evaluate_objective(objective, xk)
        rel_delta = float(torch.abs(Jk - Jprev) / torch.abs(Jprev))
        Jprev = Jk
        iterations += 1

    message = "success" if iterations < max_iter else "max_iter"
    norm_pg = float(_projected_gradient(xk, gk).norm())
    return ProjectedNewtonResult(xk, message, float(rel_delta), norm_pg, iterations)


def _evaluate_objective(objective: Objective, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, grad, hess = objective(x)
    value = value.to(torch.double)
    grad = grad.to(torch.double).reshape(-1)
    if grad.ndim != 1:
        raise ValueError("Gradient must be one-dimensional.")
    hess = hess.to(torch.double)
    if value.numel() != 1:
        raise ValueError("Objective value must be scalar.")
    if hess.ndim != 2 or hess.shape[0] != grad.numel() or hess.shape[1] != grad.numel():
        raise ValueError("Hessian has incompatible shape.")
    return value.squeeze(), grad, hess


def _projected_gradient(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    mask = (x == 0) & (grad >= 0)
    projected = grad.clone()
    projected[mask] = 0.0
    return projected


def _compute_direction(method: Method, hessian: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if method == "true_hessian":
        chol = torch.linalg.cholesky(hessian)
        return torch.cholesky_solve(grad.unsqueeze(-1), chol, upper=False).squeeze(-1)
    if method == "mod_hessian":
        try:
            chol = torch.linalg.cholesky(hessian)
            return torch.cholesky_solve(grad.unsqueeze(-1), chol, upper=False).squeeze(-1)
        except RuntimeError:
            sym_h = 0.5 * (hessian + hessian.T)
            eigvals, eigvecs = torch.linalg.eigh(sym_h)
            eigvals = torch.clamp(eigvals.abs(), min=1e-8)
            inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
            return inv @ grad
    if method == "gradient":
        return grad
    raise ValueError(f"Unknown method '{method}'.")


def _armijo_line_search(
    xk: torch.Tensor,
    grad: torch.Tensor,
    direction: torch.Tensor,
    obj_val: torch.Tensor,
    active: torch.Tensor,
    objective: Objective,
) -> float:
    sigma = 1e-4
    beta = 2.0
    alpha = beta
    it = 0
    max_iter = 15
    obj_lin = obj_val + 1.0
    zeros = torch.zeros_like(xk)
    while obj_val < obj_lin and it < max_iter:
        alpha = alpha / beta
        alpha_pk = alpha * direction
        candidate = torch.maximum(zeros, xk - alpha_pk)
        cand_val, _, _ = _evaluate_objective(objective, candidate)
        alpha_pk_adj = alpha_pk.clone()
        alpha_pk_adj[active] = xk[active] - candidate[active]
        obj_lin = cand_val + sigma * torch.dot(grad, alpha_pk_adj)
        it += 1
    if obj_val < obj_lin:
        xkh = xk.clone()
        xkh[active] = 0.0
        indices = torch.where((xkh > 0) & (direction > 0))[0]
        if indices.numel() == 0:
            return 1.0
        ratios = xk[indices] / direction[indices]
        return float(torch.min(ratios))
    return float(alpha)


__all__ = ["projected_newton", "ProjectedNewtonResult"]
