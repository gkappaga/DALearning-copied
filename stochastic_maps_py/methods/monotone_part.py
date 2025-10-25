from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from stochastic_maps_py.tools import quantiles_sorted_vector


@dataclass
class MonotonePart:
    order: int
    scaling_rbf: float = 2.0
    kappa: float = 4.0
    npoints_interp: int = 2000

    coeffs: Optional[torch.Tensor] = field(default=None, repr=False)
    centers: Optional[torch.Tensor] = field(default=None, repr=False)
    widths: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("MonotonePart requires order >= 1.")

    def ncoeff(self) -> int:
        if self.order == 1:
            return 1
        return self.order + 1

    def set_identity_function(self) -> "MonotonePart":
        self.coeffs = torch.tensor([1.0], dtype=torch.double)
        self.centers = None
        self.widths = None
        return self

    def construct_basis(self, samples: torch.Tensor) -> "MonotonePart":
        samples = _ensure_column(samples)
        if self.order == 1:
            self.centers = None
            self.widths = None
        else:
            centers, widths = self._centers_and_widths(samples, self.ncoeff())
            self.centers = centers
            self.widths = widths
        return self

    def evaluate(self, samples: torch.Tensor) -> torch.Tensor:
        self._ensure_basis(samples)
        basis = self.basis_eval(samples)
        return (basis @ self._coeffs()).reshape(-1, 1)

    def grad_x(self, samples: torch.Tensor) -> torch.Tensor:
        self._ensure_basis(samples)
        grad = self.basis_grad(samples)
        return (grad @ self._coeffs()).reshape(-1, 1)

    def basis_eval(self, samples: torch.Tensor) -> torch.Tensor:
        samples = _ensure_column(samples)
        if self.order == 1:
            return samples
        return self._eval_monotone_rbfs(samples, self.centers, self.widths)

    def basis_grad(self, samples: torch.Tensor) -> torch.Tensor:
        samples = _ensure_column(samples)
        if self.order == 1:
            return torch.ones_like(samples)
        return self._grad_monotone_rbfs(samples, self.centers, self.widths)

    def basis_hess(self, samples: torch.Tensor) -> torch.Tensor:
        samples = _ensure_column(samples)
        if self.order == 1:
            return torch.zeros_like(samples)
        return self._hess_monotone_rbfs(samples, self.centers, self.widths)

    def inverse(self, targets: torch.Tensor) -> torch.Tensor:
        if self.order == 1:
            coeff = self._coeffs()
            return targets / coeff[0]

        centers = self.centers
        widths = self.widths
        if centers is None or widths is None:
            raise ValueError("Basis must be constructed before inversion.")

        lower = centers[0] - self.kappa * widths[0]
        upper = centers[-1] + self.kappa * widths[-1]
        grid = torch.linspace(lower, upper, self.npoints_interp, dtype=torch.double)
        basis = self._eval_monotone_rbfs(grid.unsqueeze(-1), centers, widths)
        values = basis @ self._coeffs()
        return self._invert_monotone_map(targets, grid, values)

    def _coeffs(self) -> torch.Tensor:
        if self.coeffs is None:
            raise ValueError("Coefficients are not initialised.")
        return self.coeffs

    def _centers_and_widths(self, samples: torch.Tensor, ncoeff: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_samples = torch.sort(samples.squeeze(-1))[0]
        if torch.numel(sorted_samples) == 0:
            raise ValueError("Samples must be non-empty.")

        if ncoeff == 1:
            q = quantiles_sorted_vector(sorted_samples, torch.tensor([0.25, 0.5, 0.75], dtype=torch.double))
            centers = q[1:2]
            widths = torch.tensor([(q[2] - q[0]) / 2.0], dtype=torch.double)
        else:
            centers = quantiles_sorted_vector(sorted_samples, ncoeff)
            if ncoeff == 2:
                widths = (centers[1] - centers[0]) * torch.ones_like(centers)
            else:
                widths = torch.zeros_like(centers)
                widths[1:-1] = 0.5 * (centers[2:] - centers[:-2])
                widths[0] = centers[1] - centers[0]
                widths[-1] = centers[-1] - centers[-2]
        widths = self.scaling_rbf * widths
        return centers, widths

    @staticmethod
    def _eval_monotone_rbfs(samples: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        ncoeff = centers.numel()
        N = samples.shape[0]
        values = torch.zeros((N, ncoeff), dtype=torch.double)
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=torch.double))
        sqrt_two_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi, dtype=torch.double))
        delta = (samples - centers) / widths / sqrt_two
        erf_delta = torch.erf(delta)
        exp_delta = torch.exp(-delta ** 2)
        values[:, 0] = 0.5 * (
            sqrt_two * widths[0] * delta[:, 0] * (1 - erf_delta[:, 0]) - widths[0] * sqrt_two_over_pi * exp_delta[:, 0]
        )
        if ncoeff > 2:
            values[:, 1:-1] = 0.5 * (1 + erf_delta[:, 1:-1])
        values[:, -1] = 0.5 * (
            sqrt_two * widths[-1] * delta[:, -1] * (1 + erf_delta[:, -1]) + widths[-1] * sqrt_two_over_pi * exp_delta[:, -1]
        )
        return values

    @staticmethod
    def _grad_monotone_rbfs(samples: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        ncoeff = centers.numel()
        grad = torch.zeros((samples.size(0), ncoeff), dtype=torch.double)
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=torch.double))
        sqrt_two_pi = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.double))
        delta = (samples - centers) / widths / sqrt_two
        exp_delta = torch.exp(-delta ** 2)
        grad[:, 0] = 0.5 * (1 - torch.erf(delta[:, 0]))
        if ncoeff > 2:
            grad[:, 1:-1] = exp_delta[:, 1:-1] / (widths[1:-1] * sqrt_two_pi)
        grad[:, -1] = 0.5 * (1 + torch.erf(delta[:, -1]))
        return grad

    @staticmethod
    def _hess_monotone_rbfs(samples: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        ncoeff = centers.numel()
        hess = torch.zeros((samples.size(0), ncoeff), dtype=torch.double)
        sqrt_two = torch.sqrt(torch.tensor(2.0, dtype=torch.double))
        sqrt_two_pi = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.double))
        delta = (samples - centers) / widths / sqrt_two
        exp_delta = torch.exp(-delta ** 2)
        hess[:, 0] = -exp_delta[:, 0] / (widths[0] * sqrt_two_pi)
        if ncoeff > 2:
            mid = exp_delta[:, 1:-1] / (widths[1:-1] * sqrt_two_pi)
            hess[:, 1:-1] = -mid * delta[:, 1:-1] * (sqrt_two / widths[1:-1])
        hess[:, -1] = exp_delta[:, -1] / (widths[-1] * sqrt_two_pi)
        return hess

    def _ensure_basis(self, samples: torch.Tensor) -> None:
        if self.order == 1:
            return
        if self.centers is None or self.widths is None:
            if samples is None:
                raise ValueError("Samples required to construct basis.")
            self.construct_basis(samples)

    @staticmethod
    def _invert_monotone_map(points: torch.Tensor, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        points = points.reshape(-1)
        n = len(yy)
        idx_plus = torch.searchsorted(yy, points.clamp(yy[0], yy[-1]))
        idx_plus = torch.clamp(idx_plus, 1, n - 1)
        idx_min = idx_plus - 1
        y_min = yy[idx_min]
        y_max = yy[idx_plus]
        delta = (points - y_min) / (y_max - y_min)
        x_vals = (1 - delta) * xx[idx_min] + delta * xx[idx_plus]
        return x_vals


def _ensure_column(samples: torch.Tensor) -> torch.Tensor:
    samples = samples.to(torch.double)
    if samples.ndim == 1:
        samples = samples.unsqueeze(-1)
    if samples.ndim != 2 or samples.shape[1] != 1:
        raise ValueError("Samples must be a column vector.")
    return samples


__all__ = ["MonotonePart"]
