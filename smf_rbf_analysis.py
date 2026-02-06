"""
Stochastic Map Filter (SMF) - Python Port
Based on Baptista, Spantini, and Marzouk (2019)
"Coupling Techniques for Nonlinear Ensemble Filtering"

This module provides a Python implementation of the stochastic map filter
for data assimilation, ported from the original MATLAB code.
"""

import torch
import numpy as np
from scipy import stats
from typing import Optional, Tuple, Callable, Union


class MonotonePart:
    """
    Univariate nonlinear and monotone function parametrized using
    a linear term and RBFs.

    order = 1: linear
    order > 1: (order-1) RBFs + 2 erf functions
    """

    def __init__(self, d: int, order: int, scaling_rbf: float = 2.0,
                 kappa: float = 4.0, npoints_interp: int = 2000):
        if d != 1:
            raise ValueError("MonotonePart should be one-dimensional.")
        if order <= 0:
            raise ValueError("Order must be greater than 0.")

        self.d = d
        self.order = order
        self.scaling_rbf = scaling_rbf
        self.kappa = kappa
        self.npoints_interp = npoints_interp

        self.coeffs = None
        self.centers = None
        self.widths = None

    def ncoeff(self) -> int:
        """Number of coefficients in the basis."""
        if self.order == 1:
            return 1
        else:
            return (self.order - 1) + 2

    def set_id_function(self):
        """Set to identity map."""
        # NOTE: keep as CPU float by default; we move to X.device/X.dtype at evaluate time.
        self.coeffs = torch.tensor([1.0])
        self.widths = None
        self.centers = None

    def construct_basis(self, X: torch.Tensor):
        """Construct basis functions from samples."""
        if X.shape[1] != 1:
            raise ValueError("Samples should be a column vector.")

        if self.order == 1:
            self.centers = None
            self.widths = None
        else:
            self.centers, self.widths = self._centers_and_widths(X, self.ncoeff())

    def _centers_and_widths(self, X: torch.Tensor, ncoeff: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_sorted = torch.sort(X.flatten())[0]
        dev = X_sorted.device
        dt = X_sorted.dtype

        if ncoeff == 1:
            qq = torch.quantile(
                X_sorted,
                torch.tensor([0.25, 0.5, 0.75], device=dev, dtype=dt)
            )
            centers = qq[1:2]
            widths = ((qq[2] - qq[0]) / 2).reshape(1)
        else:
            quantiles = torch.linspace(0, 1, ncoeff + 2, device=dev, dtype=dt)[1:-1]
            centers = torch.quantile(X_sorted, quantiles)

            widths = torch.zeros(ncoeff, device=dev, dtype=dt)
            if ncoeff == 2:
                widths[:] = centers[1] - centers[0]
            else:
                widths[1:-1] = (centers[2:] - centers[:-2]) / 2
                widths[0] = centers[1] - centers[0]
                widths[-1] = centers[-1] - centers[-2]

        widths = self.scaling_rbf * widths
        widths = widths.clamp_min(1e-6)
        return centers, widths

    def basis_eval(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate basis functions."""
        if self.order == 1:
            return X
        else:
            return self._eval_monotone_rbfs(X, self.centers, self.widths)

    def basis_grad(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of basis functions."""
        if self.order == 1:
            return torch.ones_like(X)
        else:
            return self._grad_x_monotone_rbfs(X, self.centers, self.widths)

    def _eval_monotone_rbfs(self, X: torch.Tensor, centers: torch.Tensor,
                            widths: torch.Tensor) -> torch.Tensor:
        """Evaluate monotone RBF basis functions."""
        N = X.shape[0]
        ncoeff = len(centers)

        # Centered inputs
        deltaX = (X - centers.unsqueeze(0)) / (widths.unsqueeze(0) * np.sqrt(2))

        # Evaluate basis functions
        f = torch.zeros(N, ncoeff, device=X.device, dtype=X.dtype)

        # First basis (left boundary)
        erf_vals = torch.erf(deltaX[:, 0])
        exp_vals = torch.exp(-deltaX[:, 0] ** 2)
        f[:, 0] = 0.5 * (np.sqrt(2) * widths[0] * deltaX[:, 0] * (1 - erf_vals)
                         - widths[0] * np.sqrt(2 / np.pi) * exp_vals)

        # Middle bases
        if ncoeff > 2:
            f[:, 1:-1] = 0.5 * (1 + torch.erf(deltaX[:, 1:-1]))

        # Last basis (right boundary)
        erf_vals_end = torch.erf(deltaX[:, -1])
        exp_vals_end = torch.exp(-deltaX[:, -1] ** 2)
        f[:, -1] = 0.5 * (np.sqrt(2) * widths[-1] * deltaX[:, -1] * (1 + erf_vals_end)
                          + widths[-1] * np.sqrt(2 / np.pi) * exp_vals_end)

        return f

    def _grad_x_monotone_rbfs(self, X: torch.Tensor, centers: torch.Tensor,
                              widths: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of monotone RBF basis functions."""
        N = X.shape[0]
        ncoeff = len(centers)

        deltaX = (X - centers.unsqueeze(0)) / (widths.unsqueeze(0) * np.sqrt(2))

        df = torch.zeros(N, ncoeff, device=X.device, dtype=X.dtype)

        # First basis
        df[:, 0] = 0.5 * (1 - torch.erf(deltaX[:, 0]))

        # Middle bases
        if ncoeff > 2:
            df[:, 1:-1] = torch.exp(-deltaX[:, 1:-1] ** 2) / (
                widths[1:-1].unsqueeze(0) * np.sqrt(2 * np.pi)
            )

        # Last basis
        df[:, -1] = 0.5 * (1 + torch.erf(deltaX[:, -1]))

        return df

    def _ensure_coeffs_on(self, ref: torch.Tensor):
        if self.coeffs is None:
            raise RuntimeError("MonotonePart.coeffs is None. Did you call set_id_function() or optimize()?")

        if self.coeffs.device != ref.device or self.coeffs.dtype != ref.dtype:
            self.coeffs = self.coeffs.to(device=ref.device, dtype=ref.dtype)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the monotone function."""
        self._ensure_coeffs_on(X)
        return self.basis_eval(X) @ self.coeffs

    def grad_x(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of the monotone function."""
        self._ensure_coeffs_on(X)
        return self.basis_grad(X) @ self.coeffs

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        """Invert the monotone function using interpolation."""
        self._ensure_coeffs_on(Z)

        if self.order == 1:
            return Z / self.coeffs[0]
        else:
            # Define interpolation domain
            lbound = self.centers[0] - self.kappa * self.widths[0]
            ubound = self.centers[-1] + self.kappa * self.widths[-1]
            xx_interp = torch.linspace(
                lbound, ubound, self.npoints_interp, device=Z.device, dtype=Z.dtype
            ).unsqueeze(1)

            # Evaluate function at interpolation points
            Psi_mon = self._eval_monotone_rbfs(xx_interp, self.centers.to(Z.device), self.widths.to(Z.device))
            yy_interp = Psi_mon @ self.coeffs

            # Invert using linear interpolation
            return self._invert_1d_map_interp(Z, xx_interp.squeeze(), yy_interp)

    def _invert_1d_map_interp(self, points: torch.Tensor, xx: torch.Tensor,
                              yy: torch.Tensor) -> torch.Tensor:
        """Invert 1D monotone map using linear interpolation."""
        points_flat = points.flatten()

        # Find bracketing indices
        indices = torch.searchsorted(yy, points_flat)
        indices = torch.clamp(indices, 1, len(yy) - 1)

        ind_min = indices - 1
        ind_plus = indices

        # Linear interpolation
        denom = (yy[ind_plus] - yy[ind_min]).clamp_min(1e-12)
        delta = (points_flat - yy[ind_min]) / denom
        interp_values = (1 - delta) * xx[ind_min] + delta * xx[ind_plus]

        return interp_values.reshape(points.shape)


class NonMonotonePart:
    """
    Non-monotone function parametrized using linear functions and RBFs.

    order = 0: inactive variable
    order = 1: linear
    order >= 2: linear + RBF
    """

    def __init__(self, d: int, order: torch.Tensor, scaling_rbf: float = 2.0):
        self.d = d
        self.order = order if isinstance(order, torch.Tensor) else torch.tensor(order)
        self.scaling_rbf = scaling_rbf

        self.active_vars = torch.where(self.order > 0)[0]

        self.const_term = 0.0
        self.coeffs = [None] * d
        self.centers = [None] * d
        self.widths = [None] * d

    def ncoeff(self) -> int:
        """Total number of coefficients."""
        return int(self.order.sum().item())

    def set_zero_function(self):
        """Set to zero function."""
        self.const_term = 0.0
        total_coeffs = self.ncoeff()
        if total_coeffs > 0:
            # Keep on CPU by default; moved to X.device/X.dtype at evaluate time.
            self.coeffs = [torch.zeros(total_coeffs)]

    def construct_basis(self, X: torch.Tensor):
        """Construct basis functions from samples."""
        if X.shape[1] != self.d:
            raise ValueError("Dimension mismatch")

        for ii in self.active_vars:
            order_ii = int(self.order[ii].item())
            if order_ii == 1:
                self.centers[ii] = None
                self.widths[ii] = None
            else:
                centers, widths = self._centers_and_widths(X[:, ii:ii + 1], order_ii - 1)
                self.centers[ii] = centers
                self.widths[ii] = widths

    def _centers_and_widths(self, X: torch.Tensor, ncoeff: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute centers and widths of RBF basis using quantiles."""
        X_sorted = torch.sort(X.flatten())[0]
        dev = X_sorted.device
        dt = X_sorted.dtype

        if ncoeff == 1:
            qq = torch.quantile(
                X_sorted,
                torch.tensor([0.25, 0.5, 0.75], device=dev, dtype=dt)
            )
            centers = qq[1:2]
            widths = ((qq[2] - qq[0]) / 2).unsqueeze(0)
        else:
            quantiles = torch.linspace(0, 1, ncoeff + 2, device=dev, dtype=dt)[1:-1]
            centers = torch.quantile(X_sorted, quantiles)

            widths = torch.zeros(ncoeff, device=dev, dtype=dt)
            if ncoeff == 2:
                widths[:] = centers[1] - centers[0]
            else:
                widths[1:-1] = (centers[2:] - centers[:-2]) / 2
                widths[0] = centers[1] - centers[0]
                widths[-1] = centers[-1] - centers[-2]

        widths = self.scaling_rbf * widths
        widths = widths.clamp_min(1e-6)
        return centers, widths

    def _eval_rbf(self, X: torch.Tensor, centers: torch.Tensor,
                  widths: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian RBF basis functions."""
        return torch.exp(-((X - centers.unsqueeze(0)) / widths.unsqueeze(0)) ** 2 / 2) / \
            (widths.unsqueeze(0) * np.sqrt(2 * np.pi))

    def _grad_x_rbf(self, X: torch.Tensor, centers: torch.Tensor,
                    widths: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of Gaussian RBF basis functions."""
        r = self._eval_rbf(X, centers, widths)
        return r * (-1) * ((X - centers.unsqueeze(0)) / widths.unsqueeze(0) ** 2)

    def basis_eval(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis functions."""
        N = X.shape[0]

        # Handle case when there are no dimensions
        if self.d == 0 or X.shape[1] == 0 or self.ncoeff() == 0:
            return torch.zeros(N, 0, device=X.device, dtype=X.dtype)

        basis_eval = torch.zeros(N, self.ncoeff(), device=X.device, dtype=X.dtype)

        counter = 0
        for ii in self.active_vars:
            order_ii = int(self.order[ii].item())
            samples_ii = X[:, ii:ii + 1]

            if order_ii == 1:
                basis_eval[:, counter] = samples_ii.squeeze()
                counter += 1
            else:
                centers = self.centers[ii].to(device=X.device, dtype=X.dtype)
                widths = self.widths[ii].to(device=X.device, dtype=X.dtype)
                rbf_eval = self._eval_rbf(samples_ii, centers, widths)
                basis_eval[:, counter:counter + order_ii] = torch.cat([samples_ii, rbf_eval], dim=1)
                counter += order_ii

        return basis_eval

    def basis_grad(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of all basis functions."""
        N = X.shape[0]
        grad_eval = torch.zeros(N, self.d, self.ncoeff(), device=X.device, dtype=X.dtype)

        counter = 0
        for ii in self.active_vars:
            order_ii = int(self.order[ii].item())
            samples_ii = X[:, ii:ii + 1]

            if order_ii == 1:
                grad_eval[:, ii, counter] = 1.0
                counter += 1
            else:
                centers = self.centers[ii].to(device=X.device, dtype=X.dtype)
                widths = self.widths[ii].to(device=X.device, dtype=X.dtype)
                rbf_grad = self._grad_x_rbf(samples_ii, centers, widths)
                grad_eval[:, ii, counter] = 1.0
                grad_eval[:, ii, counter + 1:counter + order_ii] = rbf_grad
                counter += order_ii

        return grad_eval

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the non-monotone function."""
        # Handle case when there are no off-diagonal dimensions
        if self.d == 0 or X.shape[1] == 0 or self.ncoeff() == 0:
            return torch.full((X.shape[0],), float(self.const_term), device=X.device, dtype=X.dtype)

        coeffs_list = [c for c in self.coeffs if c is not None]
        if len(coeffs_list) == 0:
            return torch.full((X.shape[0],), float(self.const_term), device=X.device, dtype=X.dtype)

        coeffs_list = [c.to(device=X.device, dtype=X.dtype) for c in coeffs_list]
        coeffs_flat = torch.cat(coeffs_list)

        const = torch.tensor(float(self.const_term), device=X.device, dtype=X.dtype)
        return self.basis_eval(X) @ coeffs_flat + const

    def grad_x(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of the non-monotone function."""
        if X.shape[1] == 0 or self.ncoeff() == 0:
            return torch.zeros(X.shape[0], self.d, device=X.device, dtype=X.dtype)

        coeffs_list = [c for c in self.coeffs if c is not None]
        if len(coeffs_list) == 0:
            return torch.zeros(X.shape[0], self.d, device=X.device, dtype=X.dtype)

        coeffs_list = [c.to(device=X.device, dtype=X.dtype) for c in coeffs_list]
        coeffs_flat = torch.cat(coeffs_list)

        grad_basis = self.basis_grad(X)
        return torch.einsum('ndk,k->nd', grad_basis, coeffs_flat)


class TransportMapComponent:
    """
    Component of a triangular transport map with non-monotone off-diagonal
    and monotone diagonal parts.
    """

    def __init__(self, d: int, order: list, options: dict):
        self.d = d
        self.order = order

        # Create off-diagonal and diagonal components
        self.off_d = NonMonotonePart(d - 1, torch.tensor(order[:-1]),
                                     scaling_rbf=options.get('scalingWidths', 2.0))
        self.diag = MonotonePart(1, order[-1],
                                 scaling_rbf=options.get('scalingWidths', 2.0),
                                 kappa=options.get('kappa', 4.0),
                                 npoints_interp=options.get('npoints_interp', 2000))

        self.lambda_ = options.get('lambda', 0.0)
        self.delta = options.get('delta', 1e-8)

    def ncoeff(self) -> int:
        """Total number of coefficients."""
        return self.off_d.ncoeff() + self.diag.ncoeff()

    def set_id_comp(self):
        """Set to identity component."""
        self.off_d.set_zero_function()
        self.diag.set_id_function()

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate map component."""
        if X.shape[1] != self.d:
            raise ValueError("Dimension mismatch")
        return self.off_d.evaluate(X[:, :-1]) + self.diag.evaluate(X[:, -1:])

    def inverse(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Invert map component."""
        s_off_d = self.off_d.evaluate(X)
        return self.diag.inverse(Z - s_off_d.unsqueeze(1))

    def grad_x(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of map component."""
        grad_off = self.off_d.grad_x(X[:, :-1])
        grad_diag = self.diag.grad_x(X[:, -1:])
        return torch.cat([grad_off, grad_diag], dim=1)

    def optimize(self, X: torch.Tensor):
        """Optimize component parameters using nonlinear regression."""
        N = X.shape[0]

        # Construct and evaluate basis functions
        self.diag.construct_basis(X[:, -1:])
        psi_mon = self.diag.basis_eval(X[:, -1:])
        dpsi_mon = self.diag.basis_grad(X[:, -1:])

        # Normalize monotone basis
        mean_psi = psi_mon.mean(dim=0, keepdim=True)
        std_psi = psi_mon.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
        psi_mon = (psi_mon - mean_psi) / std_psi
        dpsi_mon = dpsi_mon / std_psi

        Q1 = None
        R = None
        mean_x = None
        std_x = None
        psi_off_d = None

        if self.d > 1:
            # Construct off-diagonal basis
            self.off_d.construct_basis(X[:, :-1])
            psi_off_d = self.off_d.basis_eval(X[:, :-1])

            # ---- FIX: avoid std() warning / NaNs when there are 0 off-diagonal columns ----
            if psi_off_d.shape[1] == 0:
                # Purely diagonal fit
                A = (psi_mon.T @ psi_mon) / N
            else:
                # Normalize off-diagonal basis
                mean_x = psi_off_d.mean(dim=0, keepdim=True)
                std_x = psi_off_d.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
                psi_off_d = (psi_off_d - mean_x) / std_x

                # QR decomposition for least squares
                lambda_reg = torch.sqrt(torch.tensor(self.lambda_, device=X.device, dtype=X.dtype))
                psi_aug = torch.cat([
                    psi_off_d,
                    lambda_reg * torch.eye(psi_off_d.shape[1], device=X.device, dtype=X.dtype)
                ], dim=0)

                Q, R = torch.linalg.qr(psi_aug)
                Q1 = Q[:N, :]

                A_sqrt = psi_mon - Q1 @ (Q1.T @ psi_mon)
                A = (A_sqrt.T @ A_sqrt) / N
        else:
            A = (psi_mon.T @ psi_mon) / N

        # Optimize diagonal component
        if self.diag.order == 1:
            g_mon = torch.sqrt(1.0 / A).squeeze()
            g_mon = g_mon.unsqueeze(0) if g_mon.dim() == 0 else g_mon
        else:
            x0 = torch.ones(self.diag.ncoeff(), device=X.device, dtype=X.dtype)
            g_mon = self._projected_newton(A, dpsi_mon, x0)

        # Compute off-diagonal coefficients
        if self.d > 1 and (Q1 is not None):
            rhs = (Q1.T @ psi_mon @ g_mon).unsqueeze(1)
            g_off = -torch.linalg.solve_triangular(R, rhs, upper=True).squeeze(1)

            # Rescale coefficients
            g_mon = g_mon / std_psi.squeeze()
            g_off = g_off / std_x.squeeze()

            const_term = -mean_psi @ g_mon - mean_x @ g_off

            # Store coefficients
            self.off_d.coeffs = [g_off]
            self.off_d.const_term = const_term.item()
        else:
            # No off-diagonal regressors active (or d==1)
            g_mon = g_mon / std_psi.squeeze()
            const_term = -mean_psi @ g_mon
            # keep a benign empty coeff tensor to avoid later cat/device issues
            self.off_d.coeffs = [torch.empty(0, device=X.device, dtype=X.dtype)]
            self.off_d.const_term = const_term.item()

        self.diag.coeffs = g_mon.to(device=X.device, dtype=X.dtype)

    def _projected_newton(self, A: torch.Tensor, dpsi_mon: torch.Tensor,
                          x0: torch.Tensor, max_iter: int = 100,
                          tol: float = 1e-6) -> torch.Tensor:
        """Projected Newton method for optimization."""
        N = dpsi_mon.shape[0]
        nbasis = A.shape[0]

        if x0.dim() != 1:
            raise ValueError(f"x0 must be 1D, got shape {x0.shape}")
        if x0.shape[0] != nbasis:
            raise ValueError(f"x0 must have length {nbasis}, got {x0.shape[0]}")

        # Add regularization
        A_reg = A + (self.lambda_ / N) * torch.eye(nbasis, device=A.device, dtype=A.dtype)
        b = self.delta * A_reg.sum(dim=1)

        x = x0.clone()

        for _ in range(max_iter):
            Ax = A_reg @ x

            dpsi_x = dpsi_mon @ x
            dpsi_sum = dpsi_mon.sum(dim=1)
            dS = (dpsi_x + self.delta * dpsi_sum).unsqueeze(1)

            # Gradient
            dpsi_dS = dpsi_mon / dS
            g = Ax - dpsi_dS.sum(dim=0) / N + b

            # Hessian
            weights = 1.0 / (dS * dS)
            weighted_dpsi = dpsi_mon * weights
            H = A_reg + (dpsi_mon.T @ weighted_dpsi) / N

            # Newton step with projection
            try:
                delta_x = -torch.linalg.solve(H, g)
            except Exception:
                delta_x = -g / (torch.diag(H) + 1e-8)

            alpha = 1.0
            x_new = torch.clamp(x + alpha * delta_x, min=0)

            if torch.norm(x_new - x) < tol:
                x = x_new
                break

            x = x_new

        return x + self.delta


class TransportMap:
    """
    Triangular transport map composed of multiple components.
    """

    def __init__(self, d: int, order: list, options: dict):
        self.d = d
        self.order = order
        self.S = [TransportMapComponent(k + 1, order[k], options) for k in range(d)]

    def optimize(self, X: torch.Tensor, non_id_comp: list):
        """Optimize non-identity components of the map."""
        for ck in non_id_comp:
            self.S[ck].optimize(X[:, :ck + 1])

    def eval_map(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate forward map."""
        Z = torch.zeros_like(X)
        for k in range(self.d):
            Z[:, k] = self.S[k].evaluate(X[:, :k + 1]).squeeze()
        return Z

    def eval_inv_map(self, Z: torch.Tensor) -> torch.Tensor:
        """Evaluate inverse map."""
        X = torch.zeros_like(Z)
        for k in range(self.d):
            X[:, k] = self.S[k].inverse(X[:, :k], Z[:, k:k + 1]).squeeze()
        return X

    def grad_x(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate gradient of map."""
        N = X.shape[0]
        dSx = torch.zeros(N, self.d, self.d, device=X.device, dtype=X.dtype)

        for k in range(self.d):
            grad_k = self.S[k].grad_x(X[:, :k + 1])
            dSx[:, k, :k + 1] = grad_k

        return dSx


class StochasticMapFilter:
    """
    Stochastic Map Filter for data assimilation using transport maps.
    """

    def __init__(self, model: dict, options: dict):
        self.model = model
        self.options = self._setup_default_options(options)

    def _setup_default_options(self, options: dict) -> dict:
        """Set default options for the filter."""
        defaults = {
            'scalingWidths': 2.0,
            'lambda': 0.0,
            'delta': 1e-8,
            'kappa': 4.0,
            'npoints_interp': max(2000, 2 * options.get('M', 100)),
            'locLik': 1,
        }

        for key, value in defaults.items():
            if key not in options:
                options[key] = value

        # Set order defaults
        order_all = options.get('order_all', 2)
        options['data_order'] = options.get('data_order', order_all)
        options['offdiag_order'] = options.get('offdiag_order', order_all)
        options['diag_order_obs'] = options.get('diag_order_obs', order_all)
        options['diag_order_unobs'] = options.get('diag_order_unobs', 1)

        return options

    def _setup_tmap_for_obs(self, data_idx: int) -> Tuple[list, list, torch.Tensor]:
        """
        Build transport map structure for assimilating a scalar observation y of state x[data_idx].

        The transport map is over the joint vector [y, x_perm] of dimension (1 + d_state),
        where x_perm is the state permuted so that the observed variable is first among the states.
        """
        d_state = self.model['d']
        dist_mat = self.options['distMat']

        # --- Build a robust state permutation: put the observed index first ---
        # Use CPU long indices for safe indexing of CUDA tensors.
        if dist_mat is None or dist_mat.numel() == 0:
            permutation_state = torch.tensor(
                [data_idx] + [i for i in range(d_state) if i != data_idx],
                device=torch.device("cpu"),
                dtype=torch.long
            )
        else:
            permutation_state = torch.argsort(dist_mat[data_idx, :]).to(device="cpu", dtype=torch.long)
            if permutation_state[0].item() != data_idx:
                rest = permutation_state[permutation_state != data_idx]
                permutation_state = torch.cat(
                    [torch.tensor([data_idx], device=torch.device("cpu"), dtype=torch.long), rest],
                    dim=0
                )

        # Permute distance matrix in the permuted state coordinates (state-only, not including y)
        dist_mat_perm = dist_mat[permutation_state, :][:, permutation_state]

        offdiag_order = int(self.options['offdiag_order'])
        offdiag_rad = int(self.options['offdiag_rad'])
        nonid_radius = int(self.options['nonId_radius'])

        # dist_to_order[r] = what order to use for off-diagonal dependence at distance r
        dist_to_order = [offdiag_order] * offdiag_rad + [0] * max(0, d_state - offdiag_rad)

        order: list = []
        non_id_comp: list = []

        # Total dimension is D = 1 + d_state: variable 0 is y; variables 1..d_state are states.
        # Component 0: y is identity (MATLAB behavior)
        order.append([1])
        # do NOT add 0 to non_id_comp

        # Component 1: observed state depends on y and itself: [y, x_obs]
        order.append([self.options['data_order'], self.options['diag_order_obs']])
        non_id_comp.append(1)

        # Components 2..d_state: remaining state variables in permuted order
        for p in range(1, d_state):
            comp_idx = p + 1  # because 0 is y, 1 is observed state

            dist_p_to_obs = dist_mat_perm[p, 0].item()

            if dist_p_to_obs <= nonid_radius:
                orders_p = [0] * (comp_idx + 1)

                # Dependence on y
                if self.options.get('locLik', 1) == 1:
                    orders_p[0] = 0
                else:
                    orders_p[0] = self.options['data_order']

                # Dependence on previous states
                for q in range(0, p):
                    dist_p_q = int(dist_mat_perm[p, q].item())
                    if dist_p_q < len(dist_to_order):
                        orders_p[1 + q] = dist_to_order[dist_p_q]

                # Diagonal order for this (unobserved) state
                orders_p[comp_idx] = self.options['diag_order_unobs']

                order.append(orders_p)
                non_id_comp.append(comp_idx)
            else:
                order.append([0] * comp_idx + [1])

        return order, non_id_comp, permutation_state

    def inflate(self, X: torch.Tensor) -> torch.Tensor:
        """Apply multiplicative inflation."""
        rho = float(self.options.get('rho', 0.0))
        if rho == 0.0:
            return X
        mean_x = X.mean(dim=0, keepdim=True)
        factor = torch.sqrt(torch.tensor(1.0 + rho, device=X.device, dtype=X.dtype))
        return (X - mean_x) * factor + mean_x

    def assimilate_scalar_obs(self, X_pr: torch.Tensor, data_idx: int, Yt: float) -> torch.Tensor:
        """Assimilate a scalar observation."""
        N, d_state = X_pr.shape
        device = X_pr.device
        dtype = X_pr.dtype

        # Build transport map for this specific observation
        order, non_id_comp, permutation = self._setup_tmap_for_obs(data_idx)

        # Create transport map (dimension is 1 + d_state)
        TM = TransportMap(d_state + 1, order, self.options)

        # Set identity components
        id_comp = set(range(TM.d)) - set(non_id_comp)
        for ck in id_comp:
            TM.S[ck].set_id_comp()

        # Apply inflation
        X_infl = self.inflate(X_pr)

        # Permute state so observed variable is first
        X_infl_perm = X_infl[:, permutation]
        X_pr_perm = X_pr[:, permutation]

        # Get observation noise
        if isinstance(self.model.get('sigma_y'), torch.Tensor):
            if self.model['sigma_y'].dim() > 0:
                sigma = self.model['sigma_y'][data_idx]
            else:
                sigma = self.model['sigma_y']
        else:
            sigma = self.model.get('sigma_y', 1.0)

        # Sample likelihood: y ~ N(x_0, sigma^2)
        Yi_infl = X_infl_perm[:, 0:1] + sigma * torch.randn(N, 1, device=device, dtype=dtype)

        # Build map input: [y, x]
        YX_infl = torch.cat([Yi_infl, X_infl_perm], dim=1)

        # Optimize transport map
        TM.optimize(YX_infl, non_id_comp)

        # Sample likelihood with un-inflated state
        Yi = X_pr_perm[:, 0:1] + sigma * torch.randn(N, 1, device=device, dtype=dtype)
        YX = torch.cat([Yi, X_pr_perm], dim=1)

        # Evaluate composed map: condition on observation
        YX_post = self._evaluate(TM, YX, Yt)

        # Extract state part
        X_post_perm = YX_post[:, 1:]

        # Inverse permutation
        X_post = torch.zeros_like(X_post_perm)
        for i in range(len(permutation)):
            X_post[:, permutation[i]] = X_post_perm[:, i]

        return X_post

    def _evaluate(self, TM: TransportMap, YX: torch.Tensor, Yt: float) -> torch.Tensor:
        """Evaluate the composed map for conditioning."""
        eta = TM.eval_map(YX)
        # MATLAB behavior: y-component is identity, so set eta_y = Yt directly.
        eta[:, 0] = float(Yt)
        return TM.eval_inv_map(eta)

    def sample_posterior(self, X_pr: torch.Tensor, Yt: torch.Tensor) -> torch.Tensor:
        """Generate posterior samples by sequentially assimilating observations."""
        data_idx = self.model['data_indices']
        n_obs = len(data_idx)

        X_post = X_pr.clone()
        for i in range(n_obs):
            X_post = self.assimilate_scalar_obs(X_post, data_idx[i], Yt[i].item())

        return X_post


def smf_transport_update(
    xf: torch.Tensor,
    y: torch.Tensor,
    H: Union[torch.Tensor, Callable],
    sigma_y: Union[float, torch.Tensor],
    distMat: Optional[torch.Tensor] = None,
    offdiag_rad: int = None,
    p_rbf: int = 2,
    diag_order: int = 2,
    nonId_radius: Optional[int] = None,
    M: Optional[int] = None,
    lambda_: float = 0.0,
    delta: float = 1e-8,
    scalingRbf: float = 2.0,
    rho: float = 0.0,
) -> torch.Tensor:
    """
    Wrapper function for SMF transport update.

    Implements scalar sequential assimilation from the paper.
    Each observation is assimilated one at a time.
    """
    B, N, d_state = xf.shape
    d_obs = y.shape[1]
    device = xf.device
    dtype = xf.dtype

    if M is None:
        M = N
    if nonId_radius is None:
        nonId_radius = d_state
    if distMat is None:
        distMat = torch.zeros(d_state, d_state, dtype=torch.int64, device=device)
    if offdiag_rad is None:
        offdiag_rad = d_state

    # Determine observed state variables
    if isinstance(H, torch.Tensor):
        H_mat = H.to(device=device, dtype=dtype)
        data_indices = []
        for i in range(d_obs):
            obs_row = H_mat[i]
            nonzero = torch.nonzero(obs_row).squeeze()
            if nonzero.numel() == 1:
                data_indices.append(nonzero.item())
            else:
                data_indices.append(nonzero[0].item())
    else:
        data_indices = list(range(d_obs))

    # Setup options
    base_options = {
        'M': M,
        'distMat': distMat,
        'order_all': p_rbf,
        'data_order': p_rbf,
        'offdiag_order': p_rbf,
        'diag_order_obs': diag_order,
        'diag_order_unobs': 1,
        'nonId_radius': nonId_radius,
        'offdiag_rad': int(offdiag_rad),
        'rho': rho,
        'lambda': lambda_,
        'delta': delta,
        'scalingWidths': scalingRbf,
    }

    # Process each batch
    xa_list = []
    for b in range(B):
        xf_b = xf[b]
        y_b = y[b]

        model = {
            'd': d_state,
            'sigma_y': sigma_y,
            'data_indices': data_indices,
        }

        # Create filter
        smf = StochasticMapFilter(model, base_options)

        # Sequential assimilation
        xa_b = xf_b.clone()
        for obs_idx in range(d_obs):
            state_idx = data_indices[obs_idx]
            xa_b = smf.assimilate_scalar_obs(xa_b, state_idx, y_b[obs_idx].item())

        xa_list.append(xa_b)

    return torch.stack(xa_list, dim=0)
