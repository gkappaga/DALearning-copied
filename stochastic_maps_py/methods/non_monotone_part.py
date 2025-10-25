from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from stochastic_maps_py.tools import quantiles_sorted_vector


@dataclass
class NonMonotonePart:
    dimension: int
    order: Sequence[int]
    scaling_rbf: float = 2.0

    const_term: float = 0.0
    coeffs: List[Optional[torch.Tensor]] = field(default_factory=list, repr=False)
    centers: List[Optional[torch.Tensor]] = field(default_factory=list, repr=False)
    widths: List[Optional[torch.Tensor]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.order = [int(o) for o in self.order]
        if len(self.order) != self.dimension:
            raise ValueError("Order length must match dimension.")
        if any(o < 0 for o in self.order):
            raise ValueError("Orders must be non-negative.")
        self.active_vars = [i for i, o in enumerate(self.order) if o > 0]
        self.coeffs = [None for _ in range(self.dimension)]
        self.centers = [None for _ in range(self.dimension)]
        self.widths = [None for _ in range(self.dimension)]

    def ncoeff(self) -> int:
        return int(sum(self.order))

    def set_zero_function(self) -> "NonMonotonePart":
        self.const_term = 0.0
        if self.ncoeff() > 0:
            zeros = torch.zeros(self.ncoeff(), dtype=torch.double)
            self._assign_flat_coeffs(zeros)
        return self

    def construct_basis(self, samples: torch.Tensor) -> "NonMonotonePart":
        samples = samples.to(torch.double)
        if samples.ndim != 2 or samples.shape[1] != self.dimension:
            raise ValueError("Samples must have shape (N, dimension).")
        for idx in self.active_vars:
            order_i = self.order[idx]
            if order_i > 1:
                centers, widths = self._centers_and_widths(samples[:, idx], order_i - 1)
                self.centers[idx] = centers
                self.widths[idx] = widths
            else:
                self.centers[idx] = None
                self.widths[idx] = None
        return self

    def basis_eval(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.double)
        pieces = []
        for idx in self.active_vars:
            col = samples[:, idx:idx + 1]
            order_i = self.order[idx]
            if order_i == 1:
                pieces.append(col)
            else:
                centers = self.centers[idx]
                widths = self.widths[idx]
                rbfs = self._eval_rbf(col, centers, widths)
                pieces.append(torch.cat([col, rbfs], dim=1))
        if not pieces:
            return torch.zeros(samples.size(0), 0, dtype=torch.double)
        return torch.cat(pieces, dim=1)

    def basis_grad(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.double)
        N = samples.size(0)
        gradients = torch.zeros((N, self.dimension, self.ncoeff()), dtype=torch.double)
        counter = 0
        for idx in self.active_vars:
            order_i = self.order[idx]
            if order_i == 1:
                gradients[:, idx, counter] = 1.0
                counter += 1
            else:
                centers = self.centers[idx]
                widths = self.widths[idx]
                grad = self._grad_rbf(samples[:, idx:idx + 1], centers, widths)
                gradients[:, idx, counter:counter + order_i] = torch.cat(
                    [torch.ones((N, 1), dtype=torch.double), grad], dim=1
                )
                counter += order_i
        return gradients

    def basis_hess(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.double)
        N = samples.size(0)
        hessians = torch.zeros((N, self.dimension, self.dimension, self.ncoeff()), dtype=torch.double)
        counter = 0
        for idx in self.active_vars:
            order_i = self.order[idx]
            if order_i == 1:
                counter += 1
                continue
            centers = self.centers[idx]
            widths = self.widths[idx]
            hess = self._hess_rbf(samples[:, idx:idx + 1], centers, widths)
            hessians[:, idx, idx, counter:counter + order_i] = torch.cat(
                [torch.zeros((N, 1), dtype=torch.double), hess], dim=1
            )
            counter += order_i
        return hessians

    def evaluate(self, samples: torch.Tensor) -> torch.Tensor:
        if self.ncoeff() == 0:
            return torch.full((samples.size(0), 1), float(self.const_term), dtype=torch.double)
        self._ensure_basis(samples)
        basis = self.basis_eval(samples)
        coeffs = self._flatten_coeffs()
        return (basis @ coeffs).reshape(-1, 1) + self.const_term

    def grad_x(self, samples: torch.Tensor) -> torch.Tensor:
        if self.ncoeff() == 0:
            return torch.zeros((samples.size(0), self.dimension), dtype=torch.double)
        self._ensure_basis(samples)
        gradients = self.basis_grad(samples)
        coeffs = self._flatten_coeffs()
        coeffs_expanded = coeffs.view(1, 1, -1)
        return torch.sum(gradients * coeffs_expanded, dim=2)

    def hess_x(self, samples: torch.Tensor) -> torch.Tensor:
        if self.ncoeff() == 0:
            return torch.zeros((samples.size(0), self.dimension, self.dimension), dtype=torch.double)
        self._ensure_basis(samples)
        hessians = self.basis_hess(samples)
        coeffs = self._flatten_coeffs()
        coeffs_expanded = coeffs.view(1, 1, 1, -1)
        return torch.sum(hessians * coeffs_expanded, dim=3)

    def _flatten_coeffs(self) -> torch.Tensor:
        coeffs = []
        for idx in self.active_vars:
            c = self.coeffs[idx]
            if c is None:
                raise ValueError("Coefficients not initialised.")
            coeffs.append(c)
        if not coeffs:
            return torch.zeros(0, dtype=torch.double)
        return torch.cat(coeffs)

    def _assign_flat_coeffs(self, coeffs: torch.Tensor) -> None:
        counter = 0
        for idx in self.active_vars:
            order_i = self.order[idx]
            self.coeffs[idx] = coeffs[counter:counter + order_i]
            counter += order_i

    def _ensure_basis(self, samples: torch.Tensor) -> None:
        if self.ncoeff() == 0:
            return
        missing = any(
            (self.order[idx] > 1 and self.centers[idx] is None)
            for idx in self.active_vars
        )
        if missing:
            self.construct_basis(samples)

    def _centers_and_widths(self, samples: torch.Tensor, nrbf: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = torch.sort(samples.flatten().to(torch.double))[0]
        if nrbf == 1:
            q = quantiles_sorted_vector(samples, torch.tensor([0.25, 0.5, 0.75], dtype=torch.double))
            centers = q[1:2]
            widths = torch.tensor([(q[2] - q[0]) / 2.0], dtype=torch.double)
        else:
            centers = quantiles_sorted_vector(samples, nrbf)
            if nrbf == 2:
                widths = (centers[1] - centers[0]) * torch.ones_like(centers)
            else:
                widths = torch.zeros_like(centers)
                widths[1:-1] = 0.5 * (centers[2:] - centers[:-2])
                widths[0] = centers[1] - centers[0]
                widths[-1] = centers[-1] - centers[-2]
        widths = self.scaling_rbf * widths
        return centers, widths

    @staticmethod
    def _eval_rbf(samples: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        delta = (samples - centers) / widths
        norm = widths * torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.double))
        return torch.exp(-0.5 * delta ** 2) / norm

    @staticmethod
    def _grad_rbf(samples: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        rbf = NonMonotonePart._eval_rbf(samples, centers, widths)
        return rbf * (-(samples - centers) / (widths ** 2))

    @staticmethod
    def _hess_rbf(samples: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        rbf = NonMonotonePart._eval_rbf(samples, centers, widths)
        delta = (samples - centers) / (widths ** 2)
        return rbf * (delta ** 2 - 1 / (widths ** 2))


__all__ = ["NonMonotonePart"]
