from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch

from stochastic_maps_py.methods.monotone_part import MonotonePart
from stochastic_maps_py.methods.non_monotone_part import NonMonotonePart
from stochastic_maps_py.tools import projected_newton


@dataclass
class TransportMapComponent:
    dimension: int
    order: Sequence[int]
    options: Dict[str, float]

    def __post_init__(self) -> None:
        if len(self.order) != self.dimension:
            raise ValueError("Order must have length equal to component dimension.")
        self.order = list(int(o) for o in self.order)
        self.lambda_ = float(self.options.get("lambda", 0.0))
        self.delta = float(self.options.get("delta", 1e-8))

        scaling = float(self.options.get("scalingWidths", 2.0))
        npoints = int(self.options.get("npoints_interp", 2000))
        kappa = float(self.options.get("kappa", 4.0))

        offd_order = self.order[:-1]
        diag_order = self.order[-1]

        self.OffD = NonMonotonePart(self.dimension - 1, offd_order, scaling_rbf=scaling)
        self.Diag = MonotonePart(diag_order, scaling_rbf=scaling, npoints_interp=npoints, kappa=kappa)

    def ncoeff(self) -> int:
        return self.OffD.ncoeff() + self.Diag.ncoeff()

    def set_identity_component(self) -> "TransportMapComponent":
        self.OffD.set_zero_function()
        self.Diag.set_identity_function()
        return self

    def evaluate(self, samples: torch.Tensor) -> torch.Tensor:
        self._check_input(samples)
        off_diag = self.OffD.evaluate(samples[:, :-1]) if self.OffD.ncoeff() > 0 else torch.zeros(
            samples.size(0), 1, dtype=torch.double
        )
        diag = self.Diag.evaluate(samples[:, -1:])
        return (off_diag + diag).reshape(-1)

    def inverse(self, prev_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prev_samples.ndim != 2 or prev_samples.shape[1] != self.dimension - 1:
            raise ValueError("prev_samples has incompatible shape.")
        if target.ndim != 1:
            target = target.reshape(-1)
        if self.OffD.ncoeff() > 0:
            off_diag = self.OffD.evaluate(prev_samples).reshape(-1)
        else:
            off_diag = torch.zeros_like(target)
        residual = target - off_diag
        return self.Diag.inverse(residual)

    def grad_x(self, samples: torch.Tensor) -> torch.Tensor:
        self._check_input(samples)
        gradients = torch.zeros((samples.size(0), self.dimension), dtype=torch.double)
        if self.OffD.ncoeff() > 0:
            gradients[:, :-1] = self.OffD.grad_x(samples[:, :-1])
        gradients[:, -1:] = self.Diag.grad_x(samples[:, -1:])
        return gradients

    def hess_x(self, samples: torch.Tensor) -> torch.Tensor:
        self._check_input(samples)
        hessian = torch.zeros((samples.size(0), self.dimension, self.dimension), dtype=torch.double)
        if self.OffD.ncoeff() > 0:
            hessian[:, :-1, :-1] = self.OffD.hess_x(samples[:, :-1])
        diag_h = self.Diag.basis_hess(samples[:, -1:])
        hessian[:, -1, -1] = diag_h.squeeze(-1)
        return hessian

    def get_coeffs(self) -> torch.Tensor:
        if self.OffD.ncoeff() > 0:
            coeffs = self.OffD._flatten_coeffs()
        else:
            coeffs = torch.zeros(0, dtype=torch.double)
        diag_coeffs = self.Diag._coeffs()
        return torch.cat([coeffs, diag_coeffs])

    def optimize(self, samples: torch.Tensor) -> "TransportMapComponent":
        self._check_input(samples)
        N = samples.size(0)
        diag_samples = samples[:, -1:]

        self.Diag.construct_basis(diag_samples)
        Psi_mon = self.Diag.basis_eval(diag_samples)
        dPsi_mon = self.Diag.basis_grad(diag_samples)

        meanPsi = Psi_mon.mean(dim=0)
        stdPsi = Psi_mon.std(dim=0, unbiased=False).clamp_min(1e-10)
        Psi_mon_norm = (Psi_mon - meanPsi) / stdPsi
        dPsi_mon_norm = dPsi_mon / stdPsi

        offd_basis = None
        meanX = None
        stdX = None
        if self.OffD.ncoeff() > 0:
            self.OffD.construct_basis(samples[:, :-1])
            Psi_offd = self.OffD.basis_eval(samples[:, :-1])
            if Psi_offd.numel() > 0:
                meanX = Psi_offd.mean(dim=0)
                stdX = Psi_offd.std(dim=0, unbiased=False).clamp_min(1e-10)
                Psi_offd = (Psi_offd - meanX) / stdX
                offd_basis = Psi_offd

                eye = torch.eye(Psi_offd.size(1), dtype=torch.double)
                augmented = torch.vstack([Psi_offd, torch.sqrt(torch.tensor(self.lambda_, dtype=torch.double)) * eye])
                Q, R = torch.linalg.qr(augmented, mode="reduced")
                Q1 = Q[:N]
                Asqrt = Psi_mon_norm - Q1 @ (Q1.T @ Psi_mon_norm)
                A = (Asqrt.T @ Asqrt) / N
            else:
                A = (Psi_mon_norm.T @ Psi_mon_norm) / N
                offd_basis = None
        else:
            A = (Psi_mon_norm.T @ Psi_mon_norm) / N

        if self.Diag.order == 1:
            if A.numel() != 1:
                raise ValueError("Quadratic matrix should be scalar for linear diagonal component.")
            g_mon = torch.sqrt(1.0 / A)
        else:
            g_mon = self._run_projected_newton(A, dPsi_mon_norm)

        g_off = None
        if offd_basis is not None and offd_basis.size(1) > 0:
            augmented = torch.vstack([offd_basis, torch.sqrt(torch.tensor(self.lambda_, dtype=torch.double)) * torch.eye(offd_basis.size(1), dtype=torch.double)])
            Q, R = torch.linalg.qr(augmented, mode="reduced")
            Q1 = Q[:N]
            g_off = -torch.linalg.solve(R, Q1.T @ (Psi_mon_norm @ g_mon))
            g_mon = g_mon / stdPsi
            g_off = g_off / stdX
            const_term = -meanPsi @ g_mon - meanX @ g_off
        else:
            g_mon = g_mon / stdPsi
            const_term = -meanPsi @ g_mon

        if g_off is not None:
            counter = 0
            coeffs_flat = g_off
            for kk in self.OffD.active_vars:
                order_kk = self.OffD.order[kk]
                self.OffD.coeffs[kk] = coeffs_flat[counter:counter + order_kk]
                counter += order_kk
        else:
            self.OffD.set_zero_function()

        self.OffD.const_term = float(const_term)
        self.Diag.coeffs = g_mon
        return self

    def _run_projected_newton(self, A: torch.Tensor, dPsi_mon: torch.Tensor) -> torch.Tensor:
        nbasis = A.shape[0]
        A = A + (self.lambda_ / dPsi_mon.size(0)) * torch.eye(nbasis, dtype=torch.double)
        b = self.delta * A.sum(dim=1)
        x0 = torch.ones(nbasis, dtype=torch.double)

        def objective(x: torch.Tensor):
            Ax = A @ x
            dS = dPsi_mon @ x + self.delta * dPsi_mon.sum(dim=1)
            log_dS = torch.log(dS)
            dPsi_dS = dPsi_mon / dS.unsqueeze(1)
            value = 0.5 * torch.dot(x, Ax) - log_dS.mean() + torch.dot(x, b)
            grad = Ax - dPsi_dS.mean(dim=0) + b
            hess = A + (dPsi_dS.T @ dPsi_dS) / dPsi_mon.size(0)
            return value, grad, hess

        result = projected_newton(x0, objective)
        return result.x_opt + self.delta

    def _check_input(self, samples: torch.Tensor) -> None:
        if samples.ndim != 2 or samples.shape[1] != self.dimension:
            raise ValueError("Input samples have incompatible shape.")


__all__ = ["TransportMapComponent"]
