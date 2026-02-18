from __future__ import annotations

import torch
import math
import time # For timing analysis steps
from tqdm import tqdm
from localization import pairwise_distances, dist2coeff
from smf_rbf_analysis import *

# import matplotlib.pyplot as plt # Uncomment for plotting GC test or RMSEs

# ##############################################################################
# # Utility Functions
# ##############################################################################

def center_ensemble(E, rescale=False):
    """
    Centers the ensemble E along the second dimension (dim=1).

    Args:
        E (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        rescale (bool): If True, rescale anomalies for unbiased covariance estimate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Centered anomalies, ensemble mean.
    """
    # Calculate mean along dim=1
    x = torch.mean(E, dim=1, keepdims=True)
    X_centered = E - x

    if rescale:
        # Get the size of the dimension along which mean was computed
        N = E.shape[1]
        if N > 1:
            # Rescale for unbiased covariance
            X_centered *= torch.sqrt(torch.tensor(N / (N - 1), device=E.device, dtype=E.dtype))

    return X_centered, x

def apply_inflation(ensemble, inflation_factor):
    """
    Applies multiplicative inflation to ensemble anomalies.

    Args:
        ensemble (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        inflation_factor (float): Multiplicative inflation factor.

    Returns:
        torch.Tensor: Inflated ensemble (batch_size x N_particles x d_state).
    """
    if inflation_factor is None or inflation_factor == 1.0:
        return ensemble

    # center_ensemble now expects a 3D tensor and operates on dim=1
    anomalies, mean_ens = center_ensemble(ensemble, rescale=False)
    # anomalies: (B, N_particles, d_state), mean_ens: (B, 1, d_state)
    # inflation_factor is scalar, broadcasts with anomalies
    # mean_ens broadcasts correctly with (inflation_factor * anomalies)
    inflated_ensemble = mean_ens + inflation_factor * anomalies
    return inflated_ensemble

def matrix_sqrt_psd(A, tol=1e-9):
    """
    Compute the square root of symmetric positive semi-definite matrices.
    A = V S V^T. Returns V S^(1/2) V^T. Handles batched inputs.

    Args:
        A (torch.Tensor): Symmetric PSD matrix or batch of matrices (..., N, N).
        tol (float): Tolerance for eigenvalue clamping to ensure non-negativity.

    Returns:
        torch.Tensor: Matrix square root (..., N, N).
    """
    # torch.linalg.eigh handles batched inputs (e.g., batch_size x N x N)
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    # Clamp eigenvalues to be non-negative before sqrt
    eigenvalues_sqrt = torch.sqrt(torch.clamp(eigenvalues, min=tol))

    # Reconstruct the matrix square root
    # eigenvectors @ diag_matrix @ eigenvectors_transposed
    # This handles both batched (A.ndim > 2) and single (A.ndim = 2) cases correctly
    # due to how torch.diag_embed and batched matrix multiply (@) work.
    # .transpose(-2, -1) is robust for batched or non-batched.
    return eigenvectors @ torch.diag_embed(eigenvalues_sqrt) @ eigenvectors.transpose(-2, -1)

def robust_eigh(A, cond_threshold=1e6):
    """
    A truly robust wrapper for torch.linalg.eigh.
    It first checks for non-finite values in the input `A` before computing
    the condition number to prevent linalg errors.

    Input:
    - A (torch.Tensor): Batch of square matrices, shape (B, N, N).
    - cond_threshold (float): Max valid condition number.

    Output:
    - (torch.Tensor, torch.Tensor): Eigenvalues (B, N) and Eigenvectors (B, N, N).
    """
    B, N, _ = A.shape
    device = A.device
    dtype = torch.float32 if not A.is_floating_point() else A.dtype

    finite_mask = torch.all(torch.isfinite(A), dim=(-2, -1))
    cond_nums = torch.full((B,), float('inf'), device=device, dtype=dtype)

    if finite_mask.any():
        A_finite = A[finite_mask]
        cond_nums[finite_mask] = torch.linalg.cond(A_finite)

    valid_mask = cond_nums <= cond_threshold

    eigvals = torch.full((B, N), float('nan'), device=device, dtype=dtype)
    eigvecs = torch.full((B, N, N), float('nan'), device=device, dtype=dtype)

    if valid_mask.any():
        L_valid, V_valid = torch.linalg.eigh(A[valid_mask])
        eigvals[valid_mask] = L_valid
        eigvecs[valid_mask] = V_valid
    
    return eigvals, eigvecs

# ##############################################################################
# # Bootstrap Particle Filter (Analysis Step Only)
# ##############################################################################

from torch import vmap

def bootstrap_particle_filter_analysis(
    particles_forecast,      # (batch_size, N_particles, d_state)
    observation_y,           # (batch_size, d_obs) or (d_obs,)
    observation_operator,    # callable or torch.Tensor of shape (d_obs, d_state)
    sigma_y,                 # float (std dev of observation noise)
    resampling_method="multinomial",
    resample_on_cpu=False,   # bool: If True, moves weight/index calculations to CPU
    sigma_reg=None,          # float (std dev for regularization noise)
    use_half_precision=False,# bool: If True, use float16 to save memory
    max_chunk_size=5000      # int: Max particles to process at once for non-matrix operators
):
    """
    Performs an advanced batch analysis (update & resampling) for a Bootstrap Particle Filter.
    This version is optimized for a linear observation_operator provided as a matrix
    and includes a fix for float16 underflow during resampling.

    Args:
        particles_forecast (torch.Tensor): Forecasted particles of shape (batch_size, N_particles, d_state).
        observation_y (torch.Tensor): Observation tensor of shape (batch_size, d_obs) or (d_obs,).
        observation_operator (callable or torch.Tensor): The observation mapping.
                                 - If callable: a function y_pred = h(x_forecast).
                                 - If torch.Tensor: A matrix H of shape (d_obs, d_state).
        sigma_y (float): Standard deviation of the observation noise.
        resampling_method (str): Resampling method, either "multinomial" or "systematic".
        resample_on_cpu (bool): If True, computes weights and indices on CPU to conserve GPU memory.
        sigma_reg (float, optional): Std dev for regularization noise.
        use_half_precision (bool): If True, converts key tensors to float16.
        max_chunk_size (int): Max particles for vectorized call (only for callable operators).

    Returns:
        torch.Tensor: The analysis particles of shape (batch_size, N_particles, d_state).
    """
    batch_size, N_particles, d_state = particles_forecast.shape
    device = particles_forecast.device
    original_dtype = particles_forecast.dtype

    if N_particles == 0:
        return particles_forecast

    # 1. Precision Handling
    compute_dtype = torch.float16 if use_half_precision else original_dtype
    particles_forecast = particles_forecast.to(compute_dtype)
    if observation_y is not None:
        observation_y = observation_y.to(compute_dtype)

    # 2. Update (Compute Weights)
    if observation_y is not None:
        if observation_y.ndim == 1:
            obs_y_broadcastable = observation_y.view(1, 1, -1)
        elif observation_y.ndim == 2:
            obs_y_broadcastable = observation_y.unsqueeze(1)
        else:
            raise ValueError("observation_y must be a 1D or 2D tensor.")

        if isinstance(observation_operator, torch.Tensor):
            H = observation_operator.to(device=device, dtype=compute_dtype)
            if H.shape[1] != d_state:
                raise ValueError(f"Matrix shape mismatch: H requires {d_state} columns, but has {H.shape[1]}")
            y_forecast = particles_forecast @ H.T
            diff_sq = ((obs_y_broadcastable - y_forecast) / sigma_y) ** 2
            log_weights = -0.5 * torch.sum(diff_sq, dim=2)
        elif callable(observation_operator):
            log_weights = torch.zeros(batch_size, N_particles, device=device, dtype=compute_dtype)
            vectorized_op = vmap(vmap(observation_operator))
            for i in range(0, N_particles, max_chunk_size):
                chunk_end = min(i + max_chunk_size, N_particles)
                particles_chunk = particles_forecast[:, i:chunk_end, :]
                y_forecast_chunk = vectorized_op(particles_chunk)
                diff_sq_chunk = ((obs_y_broadcastable - y_forecast_chunk) / sigma_y) ** 2
                log_weights[:, i:chunk_end] = -0.5 * torch.sum(diff_sq_chunk, dim=2)
        else:
            raise TypeError("observation_operator must be a callable or a torch.Tensor")
            
        max_log_w = torch.max(log_weights, dim=1, keepdim=True)[0]
        weights_unnormalized = torch.exp(log_weights - max_log_w)
        sum_weights = torch.sum(weights_unnormalized, dim=1, keepdim=True)
        
        uniform_dist = torch.full((N_particles,), 1.0 / N_particles, device=device, dtype=compute_dtype)
        weights = uniform_dist.unsqueeze(0).expand(batch_size, -1).clone()

        good_batches_mask = (sum_weights > 1e-6).squeeze(-1)
        if good_batches_mask.any():
            normalized_w_good = weights_unnormalized[good_batches_mask] / sum_weights[good_batches_mask]
            weights[good_batches_mask] = normalized_w_good
    else:
        weights = torch.full((batch_size, N_particles), 1.0 / N_particles, device=device, dtype=compute_dtype)

    # 3. Resampling (FIX APPLIED HERE)
    # --- For numerical stability, cast weights to float32 for resampling calculations ---
    resample_device = torch.device("cpu") if resample_on_cpu else device
    resample_dtype = torch.float32 # Use float32 to prevent underflow
    
    weights_resample = weights.to(device=resample_device, dtype=resample_dtype)
    
    # Add a small epsilon to the weights to guarantee the sum is non-zero
    weights_resample += torch.finfo(resample_dtype).eps

    if resampling_method == "multinomial":
        indices = torch.multinomial(weights_resample, N_particles, replacement=True)
    elif resampling_method == "systematic":
        cdf = torch.cumsum(weights_resample, dim=1)
        cdf[:, -1] = 1.0
        u_start = torch.rand(batch_size, 1, device=resample_device, dtype=resample_dtype) / N_particles
        u_uniform_strata = torch.arange(N_particles, device=resample_device, dtype=resample_dtype) / N_particles
        u_samples = u_start + u_uniform_strata.unsqueeze(0)
        indices = torch.searchsorted(cdf, u_samples, right=True).clamp_(0, N_particles - 1)
    else:
        raise ValueError(f"Unknown resampling method: {resampling_method}")

    indices = indices.to(device=device)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    particles_analysis = particles_forecast[batch_indices, indices]

    # 4. Regularization
    if sigma_reg is not None and sigma_reg > 0:
        particles_analysis += torch.randn_like(particles_analysis) * sigma_reg

    # 5. Final Type Casting
    if use_half_precision:
        particles_analysis = particles_analysis.to(original_dtype)
        
    return particles_analysis

# ======================================================================
# Stochastic Map Filter (SMF) - PyTorch port modeled on map-filters/stochasticMaps
# ======================================================================

def _smf_apply_H(particles: torch.Tensor, observation_operator, max_chunk_size: int = 10**6):
    """
    Apply observation operator to particles (B,N,d) -> (B,N,m).

    Accepts:
      • H as a 2-D tensor with shape (m,d) OR (d,m)
      • H_fun as a callable that you already use like: H_fun(batch_v[i].unsqueeze(1))
        where batch_v[i].unsqueeze(1) has shape (N,1,d) and returns (N,1,m) or (N,m).
    """
    B, N, d = particles.shape
    device, dtype = particles.device, particles.dtype

    # --- Case 1: matrix H ---
    if isinstance(observation_operator, torch.Tensor):
        H = observation_operator.to(device=device, dtype=dtype)
        if H.dim() != 2:
            raise ValueError(f"H must be 2-D, got {H.dim()}D.")
        if H.shape[1] == d:      # H is (m, d)
            return particles @ H.t()    # (B,N,d) @ (d,m) -> (B,N,m)
        if H.shape[0] == d:      # H is (d, m)
            return particles @ H        # (B,N,d) @ (d,m) -> (B,N,m)
        raise ValueError(f"Incompatible H shape {tuple(H.shape)} for state dim d={d}.")

    # --- Case 2: callable H_fun (mirror your working code) ---
    if callable(observation_operator):
        H_fun = observation_operator
        Y_list = []
        for b in range(B):
            # pass (N,1,d) just like your working site
            yb = H_fun(particles[b].unsqueeze(1))  # expected (N,1,m) or (N,m)
            if yb.dim() == 3:
                if yb.shape[1] != 1:
                    raise ValueError(f"H_fun returned shape {tuple(yb.shape)}; expected (N,1,m).")
                yb = yb.squeeze(1)  # (N,m)
            elif yb.dim() != 2:
                raise ValueError(f"H_fun returned shape {tuple(yb.shape)}; expected (N,1,m) or (N,m).")
            Y_list.append(yb)
        return torch.stack(Y_list, dim=0)  # (B,N,m)
    raise TypeError("observation_operator must be a matrix (torch.Tensor) or a callable")

def _smf_cov_emp(X: torch.Tensor, eps: float = 1e-6):
    """Unweighted empirical covariance per batch. X: (B,N,d) -> (B,d,d)."""
    B, N, d = X.shape
    Xc = X - X.mean(1, keepdim=True)
    C = (Xc.transpose(1,2) @ Xc) / max(1, N - 1)
    eye = torch.eye(d, device=X.device, dtype=X.dtype).unsqueeze(0)
    return C + eps * eye

def _smf_mask_cov_with_dist(Sxx: torch.Tensor, distMat: torch.Tensor | None, offdiag_rad: float | None):
    """
    Optional localization: keep state-state covariances within radius defined by distMat.
    Sxx: (B,d,d); distMat: (d,d) with pairwise distances; offdiag_rad (float or None).
    """
    if distMat is None or offdiag_rad is None:
        return Sxx
    device, dtype = Sxx.device, Sxx.dtype
    M = (distMat.to(device=device).to(dtype=dtype) <= float(offdiag_rad)).to(dtype)
    diag = torch.diagonal(Sxx, dim1=-2, dim2=-1)
    S_mask = Sxx * M.unsqueeze(0)
    return S_mask - torch.diag_embed(torch.diagonal(S_mask, dim1=-2, dim2=-1)) + torch.diag_embed(diag)

# -------------------- RBF helpers --------------------


def _smf_fit_linear_KR(Z: torch.Tensor,
                       X: torch.Tensor,
                       distMat: torch.Tensor | None = None,
                       offdiag_rad: float | None = None,
                       jitter: float = 1e-6):
    """
    Batched linear triangular KR map for joint [Z, X].
    Returns parameters for S^X(z,x) = L^{-1}_{x|z} ( x - μ_{x|z} ),
    but exposes A = Σ_{z z}^{-1} Σ_{z x} instead of Σ_{z z}^{-1} / Σ_{x z}.
    """
    B, N, m = Z.shape
    d = X.shape[2]
    device, dtype = X.device, X.dtype

    mu_z = Z.mean(1)                     # (B,m)
    mu_x = X.mean(1)                     # (B,d)
    Zc = Z - mu_z.unsqueeze(1)
    Xc = X - mu_x.unsqueeze(1)

    Szz = (Zc.transpose(1,2) @ Zc) / max(1, N - 1)        # (B,m,m)
    Sxz = (Xc.transpose(1,2) @ Zc) / max(1, N - 1)        # (B,d,m)
    Sxx = (Xc.transpose(1,2) @ Xc) / max(1, N - 1)        # (B,d,d)

    # Optional covariance masking on Sxx (leave as-is for now)
    Sxx = _smf_mask_cov_with_dist(Sxx, distMat, offdiag_rad)

    # Scaled jitter (safer than constant), but you can keep your jitter if you prefer
    eye_m = torch.eye(m, device=device, dtype=dtype).unsqueeze(0)
    eye_d = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    Szz = Szz + jitter * eye_m
    Sxx = Sxx + jitter * eye_d

    # Compute A = Σ_{zz}^{-1} Σ_{zx}  via two triangular solves (no explicit inverse)
    # First, Cholesky of Szz
    Lz = torch.linalg.cholesky(Szz)                          # (B,m,m)
    # Solve Lz * T = Sxz^T  -> T = Lz^{-1} Sxz^T
    T  = torch.linalg.solve_triangular(Lz, Sxz.transpose(1,2), upper=False)   # (B,m,d)
    # Solve Lz^T * A = T   -> A = (Lz^T)^{-1} T
    A  = torch.linalg.solve_triangular(Lz.transpose(1,2), T, upper=True)      # (B,m,d)

    # Schur complement: Σ_{x|z} = Sxx - Sxz * A
    Sx_given_z = Sxx - Sxz @ A                                                # (B,d,d)
    L = torch.linalg.cholesky(Sx_given_z)                                     # (B,d,d)

    return mu_z, mu_x, A, L

def _chol_spd(C: torch.Tensor, max_tries: int = 12):
    """
    Robust batch Cholesky: symmetric, try cholesky_ex; if it fails,
    add eps * mean(diag) I, with eps escalating 1e-12,1e-11,... until it works.
    """
    # Symmetrize first
    C = C + C.transpose(-1, -2)
    C.mul_(0.5)  # stays in C.dtype
    L, info = torch.linalg.cholesky_ex(C)
    if (info == 0).all():
        return L

    B, d, _ = C.shape
    I = torch.eye(d, device=C.device, dtype=C.dtype).unsqueeze(0)
    diag_mean = C.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B,1,1)
    eps = 1e-12
    C_work = C.clone()
    for _ in range(max_tries):
        bad = (info > 0)
        if not bad.any():
            break
        C_work[bad] = C[bad] + eps * diag_mean[bad] * I[:1]
        L, info = torch.linalg.cholesky_ex(0.5*(C_work + C_work.transpose(-1, -2)))
        eps *= 10.0
    # final assert—if it still fails, something else is wrong
    if (info > 0).any():
        raise RuntimeError("SMF: SPD repair failed after retries.")
    return L


def _smf_fit_linear_KR_perk(Z: torch.Tensor, X: torch.Tensor,
                       distMat: torch.Tensor | None = None,
                       offdiag_rad: float | None = None,
                       jitter: float = 1e-6):
    B, N, m = Z.shape
    d = X.shape[-1]

    mu_z = Z.mean(1)
    mu_x = X.mean(1)
    Zc = Z - mu_z.unsqueeze(1)
    Xc = X - mu_x.unsqueeze(1)

    A = torch.zeros(B, m, d, device=Z.device, dtype=Z.dtype)
    Xhat = torch.zeros(B, N, d, device=Z.device, dtype=Z.dtype)

    # (Optional) QR-based LS for better numerics
    for k in range(d):
        mk = min(k+1, m)
        Zk = Zc[:, :, :mk]             # (B,N,mk)
        xk = Xc[:, :, k:k+1]           # (B,N,1)

        # batched QR: Zk = Q R  (Q orthonormal columns, R upper-triangular)
        Q, R = torch.linalg.qr(Zk, mode='reduced')         # (B,N,mk), (B,mk,mk)
        # beta = R^{-1} Q^T xk
        rhs = torch.matmul(Q.transpose(1,2), xk)           # (B,mk,1)
        beta = torch.linalg.solve_triangular(R, rhs, upper=True)  # (B,mk,1)

        A[:, :mk, k] = beta.squeeze(-1)
        Xhat[..., k:k+1] = Zk @ beta

    # Residual covariance (exact for linear)
    Rres = Xc - Xhat
    Sx_given_z = (Rres.transpose(1,2) @ Rres) / max(1, N-1)
    # Robust Cholesky (adds tiny trace-scaled jitter only if needed)
    L = _chol_spd(Sx_given_z)
    # L = torch.linalg.cholesky(Sx_given_z)

    return mu_z, mu_x, A, L


def _smf_forward_linear_KR(mu_z, mu_x, A, L, z, x):
    """
    u = L^{-1} ( x - μ_{x|z} ), batched over B.
    z: (B,N,m) or (B,m); x: (B,N,d) -> u: (B,N,d)
    """
    # Ensure z has (B,N,m)
    if z.dim() == 2:
        z = z.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)  # (B,N,m)

    zc = z - mu_z.unsqueeze(1)                                 # (B,N,m)
    mu_x_given_z = mu_x.unsqueeze(1) + zc @ A                  # (B,N,d)

    y = (x - mu_x_given_z).transpose(1, 2)                     # (B,d,N)
    u = torch.linalg.solve_triangular(L, y, upper=False).transpose(1, 2)
    return u


def _smf_inverse_linear_KR(mu_z, mu_x, A, L, z_star, u):
    """
    x = μ_{x|z*} + L u, batched.
    z_star: (B,m) or (B,N,m); u: (B,N,d) -> x: (B,N,d)
    """
    if z_star.dim() == 2:
        z_star = z_star.unsqueeze(1).expand(u.shape[0], u.shape[1], -1)  # (B,N,m)

    zc = z_star - mu_z.unsqueeze(1)                         # (B,N,m)
    mu_x_given_zstar = mu_x.unsqueeze(1) + zc @ A           # (B,N,d)
    return mu_x_given_zstar + (u @ L.transpose(1, 2))

class SMFLinearKR:
    """
    Transport map container for the SMF linear KR map.
    Holds parameters and exposes forward/inverse like a 'transform'.
    Shapes are batched by B.
    """
    def __init__(self, mu_z, mu_x, A, L, nonId_radius=None, meta=None):
        self.kind = "smf_linear_kr"
        self.mu_z = mu_z              # (B, m)
        self.mu_x = mu_x              # (B, d)
        self.A = A                    # (B, m, d)
        self.L = L                    # (B, d, d)
        self.nonId_radius = nonId_radius
        self.meta = {} if meta is None else meta

    @torch.no_grad()
    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        u = S^X(z,x) = L^{-1} (x - μ_{x|z})
        z: (B, m) or (B, N, m); x: (B, N, d) -> u: (B, N, d)
        """
        mu_z, mu_x, A, L = self.mu_z, self.mu_x, self.A, self.L
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)  # (B,N,m)
        zc = z - mu_z.unsqueeze(1)
        mu_x_given_z = mu_x.unsqueeze(1) + zc @ A
        y = (x - mu_x_given_z).transpose(1,2)                # (B,d,N)
        u = torch.linalg.solve_triangular(L, y, upper=False).transpose(1, 2)
        return u

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x = (S^X(z,·))^{-1}(u) = μ_{x|z} + L u
        z: (B, m) or (B, N, m); u: (B, N, d) -> x: (B, N, d)
        """
        mu_z, mu_x, A, L = self.mu_z, self.mu_x, self.A, self.L
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(u.shape[0], u.shape[1], -1)
        zc = z - mu_z.unsqueeze(1)
        mu_x_given_z = mu_x.unsqueeze(1) + zc @ A
        return mu_x_given_z + (u @ L.transpose(1,2))

# ======================================================================
# RESTORE-ORIGINAL RBF PATH (matches what you pasted before)
# - Keeps your original behavior: used = range(min(k+1, m))
# - Fixed N(0,1) quantile centers + fixed widths
# - include_y adds an extra "y" block (can double-count z0, as before)
# - DOES NOT TOUCH ANY LINEAR CODE
#
# Drop this block ABOVE stochastic_map_filter_analysis, and make sure your
# p_rbf != 0 branch calls _smf_fit_separable_linear_rbf + _predict... exactly
# like in your pasted code.
# ======================================================================

import torch
from torch.distributions.normal import Normal

_STD_FLOOR = 1e-8


def _rbf_features(z_std: torch.Tensor, centers_std: torch.Tensor, widths_std: torch.Tensor):
    """
    RBF features on ALREADY-standardized scalar input.
    z_std: (B,N)
    centers_std: (B,p)
    widths_std: (B,p)
    returns: (B,N,p) with zero-mean columns (as in your original)
    """
    xc = z_std.unsqueeze(-1) - centers_std.unsqueeze(1)          # (B,N,p)
    s  = widths_std.unsqueeze(1).clamp_min(1e-6)                 # (B,1,p)
    u  = xc / s
    Phi = torch.exp(-0.5 * u * u)                                # (B,N,p)
    Phi = Phi - Phi.mean(dim=1, keepdim=True)                    # zero-mean per feature
    return Phi


class SeparableLinRBFParams:
    """
    Stores the fitted separable regression parameters.

    betas: list length d, each element is {'y': (B,q) or None, 'i': {i: (B,q)}}
    centers_std: {'y': (B,p) or None, 'i': {i: (B,p)}}
    widths_std:  {'y': (B,p) or None, 'i': {i: (B,p)}}
    mu_z_i/std_z_i: dict i -> (B,) standardization stats for each observed component
    """
    def __init__(
        self,
        *,
        betas,
        centers_std,
        widths_std,
        mu_z_i,
        std_z_i,
        include_y: bool,
        p_rbf: int
    ):
        self.betas = betas
        self.centers_std = centers_std
        self.widths_std = widths_std
        self.mu_z_i = mu_z_i
        self.std_z_i = std_z_i
        self.include_y = bool(include_y)
        self.p_rbf = int(p_rbf)


def _predict_mu_x_given_z_separable(Z: torch.Tensor, mu_x: torch.Tensor, params: SeparableLinRBFParams):
    """
    Predict μ̂_{x|z} using stored standardization stats + separable blocks.
    Z: (B,N,m) RAW observations (not centered/standardized)
    mu_x: (B,d)
    returns: (B,N,d)
    """
    B, N, m = Z.shape
    d = mu_x.shape[-1]
    device, dtype = Z.device, Z.dtype

    # Standardize Z using stored stats (only dims that exist)
    Z_std = torch.zeros_like(Z)
    for i in range(m):
        mu_i = params.mu_z_i[i].to(device=device, dtype=dtype).unsqueeze(1)   # (B,1)
        sd_i = params.std_z_i[i].to(device=device, dtype=dtype).unsqueeze(1).clamp_min(_STD_FLOOR)
        Z_std[..., i] = (Z[..., i] - mu_i) / sd_i

    Xhat = torch.zeros(B, N, d, device=device, dtype=dtype)
    q_block = 1 + params.p_rbf

    for k in range(d):
        used = list(range(min(k + 1, m)))  # <-- ORIGINAL behavior
        mats = []

        # Optional u0^k(y) where y is first observation component
        if params.include_y and m > 0:
            y_std = Z_std[..., 0]                          # (B,N)
            col = y_std.unsqueeze(-1)                      # (B,N,1)
            if params.p_rbf > 0:
                Phi_y = _rbf_features(y_std, params.centers_std['y'], params.widths_std['y'])  # (B,N,p)
                col = torch.cat([col, Phi_y], dim=-1)      # (B,N,1+p)
            mats.append(col)

        # u_i^k(z_i) terms
        for i in used:
            zi_std = Z_std[..., i]
            col = zi_std.unsqueeze(-1)
            if params.p_rbf > 0:
                Phi_i = _rbf_features(zi_std, params.centers_std['i'][i], params.widths_std['i'][i])
                col = torch.cat([col, Phi_i], dim=-1)
            mats.append(col)

        if not mats:
            Xhat[..., k] = 0.0
            continue

        Phi_k = torch.cat(mats, dim=-1)                    # (B,N,q_k)

        # Rebuild beta blocks in the same order used to build Phi_k
        beta_blocks = []
        if params.include_y and m > 0:
            beta_blocks.append(params.betas[k]['y'].unsqueeze(-1))            # (B,q_block,1)
        for i in used:
            beta_blocks.append(params.betas[k]['i'][i].unsqueeze(-1))         # (B,q_block,1)

        beta_k = torch.cat(beta_blocks, dim=1)              # (B,q_k,1)
        Xhat[..., k:k+1] = Phi_k @ beta_k                   # (B,N,1)

    return Xhat + mu_x.unsqueeze(1)


def _smf_fit_separable_linear_rbf(
    Z: torch.Tensor,   # (B,N,m) synthetic obs (RAW)
    X: torch.Tensor,   # (B,N,d) state (RAW)
    p_rbf: int,        # 1 or 2
    include_y: bool = False
):
    """
    RESTORED version: matches your pasted code.
    - Standardize Z once using per-dim mean/std
    - Fixed centers from N(0,1) quantiles
    - Fixed widths [1.0] or [0.8,1.2]
    - Fit each x_k on features of used z dims (min(k+1,m)) and optional y-term
    - L from residual covariance
    """
    p_rbf = int(p_rbf)
    assert p_rbf in (1, 2), "p_rbf must be 1 or 2"
    B, N, m = Z.shape
    d = X.shape[-1]
    device, dtype = X.device, X.dtype

    # Output mean & center
    mu_x = X.mean(1)                 # (B,d)
    Xc   = X - mu_x.unsqueeze(1)     # (B,N,d)

    # 1) stats from raw Z
    mu_z_i, std_z_i = {}, {}
    for i in range(m):
        zi = Z[..., i]
        mu_z_i[i]  = zi.mean(dim=1)                                      # (B,)
        std_z_i[i] = zi.std(dim=1, unbiased=True).clamp_min(1e-6)        # (B,)

    # 2) standardize all Z
    Z_std = torch.zeros_like(Z)
    for i in range(m):
        Z_std[..., i] = (Z[..., i] - mu_z_i[i].unsqueeze(1)) / std_z_i[i].unsqueeze(1)

    # 3) fixed centers/widths in standardized space
    centers_std = {'y': None, 'i': {}}
    widths_std  = {'y': None, 'i': {}}

    if p_rbf > 0:
        q_vals = torch.arange(1, p_rbf+1, device=device, dtype=dtype) / (p_rbf + 1.0)
        fixed_centers = Normal(0.0, 1.0).icdf(q_vals)  # (p_rbf,)

        if p_rbf == 1:
            fixed_widths = torch.tensor([1.0], device=device, dtype=dtype)
        else:
            fixed_widths = torch.tensor([0.8, 1.2], device=device, dtype=dtype)

        centers_batch = fixed_centers.unsqueeze(0).expand(B, -1)  # (B,p_rbf)
        widths_batch  = fixed_widths.unsqueeze(0).expand(B, -1)   # (B,p_rbf)

        if include_y and m > 0:
            centers_std['y'] = centers_batch
            widths_std['y']  = widths_batch

        for i in range(m):
            centers_std['i'][i] = centers_batch
            widths_std['i'][i]  = widths_batch

    # 4) fit each k
    betas = []
    Xhat = torch.zeros(B, N, d, device=device, dtype=dtype)

    for k in range(d):
        used = list(range(min(k + 1, m)))  # <-- ORIGINAL behavior
        mats = []

        if include_y and m > 0:
            y_std = Z_std[..., 0]
            col = y_std.unsqueeze(-1)
            if p_rbf > 0:
                col = torch.cat([col, _rbf_features(y_std, centers_std['y'], widths_std['y'])], dim=-1)
            mats.append(col)

        for i in used:
            zi_std = Z_std[..., i]
            col = zi_std.unsqueeze(-1)
            if p_rbf > 0:
                col = torch.cat([col, _rbf_features(zi_std, centers_std['i'][i], widths_std['i'][i])], dim=-1)
            mats.append(col)

        if mats:
            Phi_k = torch.cat(mats, dim=-1)  # (B,N,q_k)
        else:
            Phi_k = torch.zeros(B, N, 0, device=device, dtype=dtype)

        xk = Xc[..., k:k+1]  # (B,N,1)

        if Phi_k.shape[-1] > 0:
            G = Phi_k.transpose(1, 2) @ Phi_k
            ridge = 1e-5 * torch.eye(G.shape[-1], device=device, dtype=dtype).unsqueeze(0)
            beta_k = torch.linalg.solve(G + ridge, Phi_k.transpose(1, 2) @ xk)  # (B,q_k,1)
        else:
            beta_k = torch.zeros(B, 0, 1, device=device, dtype=dtype)

        # store in blocks
        rec = {'y': None, 'i': {}}
        offs = 0
        q_block = 1 + p_rbf

        if include_y and m > 0:
            rec['y'] = beta_k[:, offs:offs+q_block, 0]
            offs += q_block

        for i in used:
            rec['i'][i] = beta_k[:, offs:offs+q_block, 0]
            offs += q_block

        betas.append(rec)

        if Phi_k.shape[-1] > 0:
            Xhat[..., k:k+1] = Phi_k @ beta_k

    # 5) residual covariance -> L
    R = Xc - Xhat
    Sx_given_z = (R.transpose(1, 2) @ R) / max(1, N - 1)
    Sx_given_z = 0.5 * (Sx_given_z + Sx_given_z.transpose(1, 2))
    L = _chol_spd(Sx_given_z)  # uses your existing robust chol

    params = SeparableLinRBFParams(
        betas=betas,
        centers_std=centers_std,
        widths_std=widths_std,
        mu_z_i=mu_z_i,
        std_z_i=std_z_i,
        include_y=include_y,
        p_rbf=p_rbf
    )
    return mu_x, params, L


class SMFSeparableKR:
    """
    Separable/triangular map with linear+RBF component functions.
    forward: u = L^{-1}(x - μ̂_x|z)
    inverse: x = μ̂_x|z* + L u
    """
    def __init__(self, mu_x, params: SeparableLinRBFParams, L, nonId_radius=None, meta=None):
        self.kind = "smf_separable_rbf"
        self.mu_x = mu_x          # (B,d)
        self.params = params
        self.L = L                # (B,d,d)
        self.nonId_radius = nonId_radius
        self.meta = {} if meta is None else meta

    @torch.no_grad()
    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)
        mu_x_given_z = _predict_mu_x_given_z_separable(z, self.mu_x, self.params)
        y = (x - mu_x_given_z).transpose(1, 2)
        return torch.linalg.solve_triangular(self.L, y, upper=False).transpose(1, 2)

    @torch.no_grad()
    def inverse(self, z_star: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if z_star.dim() == 2:
            z_star = z_star.unsqueeze(1).expand(u.shape[0], u.shape[1], -1)
        mu_x_given_z = _predict_mu_x_given_z_separable(z_star, self.mu_x, self.params)
        return mu_x_given_z + (u @ self.L.transpose(1, 2))


# ======================================================================
# In stochastic_map_filter_analysis, your RBF branch should be the SAME as before:
#
# else:
#     mu_x, params, L = _smf_fit_separable_linear_rbf(
#         Zs, X[:, :, :K], p_rbf=p_rbf, include_y=include_y_in_bias
#     )
#
#     mu_x_given_z_fore = _predict_mu_x_given_z_separable(Zs, mu_x, params)
#     Y = (X[:, :, :K] - mu_x_given_z_fore).transpose(1, 2)
#     U = torch.linalg.solve_triangular(L, Y, upper=False).transpose(1, 2)
#
#     z_star_bn = y_star.unsqueeze(1).expand(U.shape[0], U.shape[1], -1)
#     mu_x_given_z_star = _predict_mu_x_given_z_separable(z_star_bn, mu_x, params)
#     X_firstK = mu_x_given_z_star + (U @ L.transpose(1, 2))
#
#     smf_map = SMFSeparableKR(mu_x=mu_x, params=params, L=L,
#                              nonId_radius=K, meta={"p_rbf": p_rbf, "rho": rho})
# ======================================================================



def stochastic_map_filter_analysis(
    particles_forecast: torch.Tensor,      # (B, N, d)
    observation_y: torch.Tensor,           # (B, m) or (m,)
    observation_operator,                  # H (m,d) or callable (B,N,d)->(B,N,m)
    sigma_y,
    *,
    # ---- options matching the repo README ----
    M: int | None = None,                  # number of samples; default uses N
    distMat: torch.Tensor | None = None,   # (d,d) pairwise state distances
    order_all: int = 1,                    # polynomial order; we implement 1 (linear)
    nonId_radius: int | None = None,       # # leading dims to transform; rest identity
    offdiag_rad: float = 0,      # localization radius for state-state cov
    rho: float = 0.0,                      # ensemble inflation on forecast
    jitter: float = 1e-6,                  # numerical jitter
    p_rbf: int = 0,                 # NEW: 0 = linear, 1 or 2 = linear+RBFs
    include_y_in_bias: bool = False # NEW: include u0^k(y) term
):
    """
    One SMF analysis step (no particle resampling). Mirrors
    SM = StochasticMapFilter(model, options); filter = seq_assimilation(..., @SM.sample_posterior)
    with options M, distMat, order_all, nonId_radius, offdiag_rad, rho. :contentReference[oaicite:1]{index=1}
    """
    smf_map = None  # will hold the resulting map
    X = particles_forecast
    X = X.to(torch.float32)
    observation_y = observation_y.to(torch.float32)
    B, N, d = X.shape
    device, dtype = X.device, X.dtype

    # 1) optional inflation (rho)
    if rho and float(rho) != 0.0:
        mean = X.mean(1, keepdim=True)
        X = mean + (1.0 + float(rho)) * (X - mean)

    # 2) synthetic observations Z ~ p(z|x); Gaussian noise with std sigma_y
    Yf = _smf_apply_H(X, observation_operator)   
    m = Yf.shape[-1]

    if not hasattr(stochastic_map_filter_analysis, "_printed_shape_info"):
        stochastic_map_filter_analysis._printed_shape_info = True
        print(f"[SMF] shapes: B={B}, N={N}, d={d}, m={m},  N/m={N/m:.1f}, N/d={N/d:.1f}")
        if m == d:
            print("[SMF] Full observation detected (m == d).")
        # Optional: gentle warning if N is small for given m
        if N / m < 20:
            print(f"[SMF][warn] N/m={N/m:.1f} is small; linear KR covariances may be noisy without localization.") 
    if isinstance(sigma_y, torch.Tensor):
        sig = sigma_y.to(device=device, dtype=dtype)
        sig = sig.view(B, 1, -1) if sig.ndim > 0 else sig.view(1,1,1)
    else:
        sig = torch.as_tensor(sigma_y, device=device, dtype=dtype).view(1,1,1)
    eps = torch.randn_like(Yf)
    eps = eps - eps.mean(dim=1, keepdim=True)     # zero-mean across particles per (B,·,m)

    Zs  = Yf + eps * sig                           # std, not variance
    

    # 3) map scope (nonId_radius)
    K = d if (nonId_radius is None) else int(nonId_radius)
    K = max(0, min(d, K))
    # 4) fit linear KR on first K dims; 5) push through inverse at y*
    if K > 0:
        y_star = observation_y if observation_y.ndim == 2 else observation_y.unsqueeze(0).expand(B, -1)

        if p_rbf == 0:
            # --- linear KR path (your existing code) ---
            mu_z, mu_x, A, L = _smf_fit_linear_KR_perk(
                Zs, X[:, :, :K],
                distMat=(None if distMat is None else distMat[:K, :K]),
                offdiag_rad=offdiag_rad, jitter=jitter
            )
            U = _smf_forward_linear_KR(mu_z, mu_x, A, L, Zs, X[:, :, :K])
            X_firstK = _smf_inverse_linear_KR(mu_z, mu_x, A, L, y_star, U)
            # with torch.no_grad():
            #     U = _smf_forward_linear_KR(mu_z, mu_x, A, L, Zs, X[:, :, :K])
            #     Xrec = _smf_inverse_linear_KR(mu_z, mu_x, A, L, Zs, U)   # <-- Zs again
            #     print("round-trip max abs:", (Xrec - X[:, :, :K]).abs().max().item())
            smf_map = SMFLinearKR(mu_z=mu_z, mu_x=mu_x, A=A, L=L, nonId_radius=K,
                                meta={"p_rbf": 0, "order_all": order_all, "rho": rho})

        else:
            # --- separable linear + RBF path (paper’s “linear + p RBFs”) ---
            # --- separable linear + RBF path ---
                # mu_x, params, L = _smf_fit_separable_linear_rbf(
                #     Zs, X[:, :, :K], p_rbf=p_rbf, include_y=include_y_in_bias
                # )
            
                # mu_x_given_z_fore = _predict_mu_x_given_z_separable(Zs, mu_x, params)
                # Y = (X[:, :, :K] - mu_x_given_z_fore).transpose(1, 2)
                # U = torch.linalg.solve_triangular(L, Y, upper=False).transpose(1, 2)
            
                # z_star_bn = y_star.unsqueeze(1).expand(U.shape[0], U.shape[1], -1)
                # mu_x_given_z_star = _predict_mu_x_given_z_separable(z_star_bn, mu_x, params)
                # X_firstK = mu_x_given_z_star + (U @ L.transpose(1, 2))
            
                # smf_map = SMFSeparableKR(mu_x=mu_x, params=params, L=L,
                #                          nonId_radius=K, meta={"p_rbf": p_rbf, "rho": rho})
                
                y_star = observation_y if observation_y.ndim == 2 else observation_y.unsqueeze(0).expand(B, -1)

                # Wrap observation operator so it matches StochasticMapFilterPy._apply_H expectations:
                # input (N,d) -> output (N,m)
                if isinstance(observation_operator, torch.Tensor):
                    H_for_smf = observation_operator  # matrix path works directly
                else:
                    H_fun = observation_operator

                    _target_device = None
                    _target_dtype = None

                    # Best-effort: if H_fun has attributes that are tensors (e.g., .proj, .H, etc.)
                    for name in ["proj", "H", "W", "weight", "matrix"]:
                        if hasattr(H_fun, name) and isinstance(getattr(H_fun, name), torch.Tensor):
                            t = getattr(H_fun, name)
                            _target_device = t.device
                            _target_dtype = t.dtype
                            break

                    # Fallback to float32 on same device as X if we can't infer
                    def H_for_smf(Xn: torch.Tensor) -> torch.Tensor:
                        # Xn: (N,d) coming from SMF code (likely float64)
                        X_in = Xn
                        if _target_device is not None:
                            X_in = X_in.to(device=_target_device)
                        else:
                            X_in = X_in.to(device=Xn.device)

                        if _target_dtype is not None:
                            X_in = X_in.to(dtype=_target_dtype)
                        else:
                            X_in = X_in.to(dtype=torch.float32)

                        # Call user H_fun (may expect (N,d) or (1,N,d) or (N,1,d))
                        try:
                            Y = H_fun(X_in)                      # (N,m) or (N,1,m) etc.
                        except Exception:
                            Y = H_fun(X_in.unsqueeze(0))         # (1,N,m) or (1,N,1,m)

                        if isinstance(Y, (tuple, list)):
                            Y = Y[0]

                        # Squeeze common singleton dims
                        if Y.dim() == 4 and Y.shape[0] == 1:
                            Y = Y.squeeze(0)
                        if Y.dim() == 3 and Y.shape[0] == 1:
                            Y = Y.squeeze(0)
                        if Y.dim() == 3 and Y.shape[1] == 1:
                            Y = Y.squeeze(1)

                        # IMPORTANT: return float64 for SMF internals
                        return Y.to(device=Xn.device, dtype=torch.float64)

                # sigma_y: make sure shape is (m,) if tensor
                sig_for_smf = sigma_y
                if isinstance(sig_for_smf, torch.Tensor):
                    sig_for_smf = sig_for_smf.to(device=X.device, dtype=torch.float64).flatten()

                # Call your batched wrapper on the first K dims only
                X_firstK = smf_transport_update(
                    xf=X[:, :, :K].to(torch.float64),      # (B,N,K)
                    y=y_star.to(torch.float64),            # (B,m)
                    H=H_for_smf,
                    sigma_y=sig_for_smf,
                    distMat=(None if distMat is None else distMat.to(torch.float64)),
                    offdiag_rad=float(0),
                    p_rbf=int(p_rbf),
                    diag_order=2,
                    nonId_radius=K,
                    M=M,
                    lambda_=0.001,
                    delta=1e-8,
                    scalingRbf=2.0,
                ).to(dtype=X.dtype)
                # print(X_firstK)
                # Splice back into full state
                X_analysis = X.clone()
                X_analysis[:, :, :K] = X_firstK
                X_firstK = X_analysis[:, :, :K]  # if you rely on this variable later
            # x_dtype = X.dtype
            # x_device = X.device

            # H0 = observation_operator
            # def h_float(X_in):
            #     X_in = X_in.to(dtype=torch.float32, device=x_device)
            #     out = H0(X_in)
            #     return out.to(dtype=x_dtype, device=x_device)
            
            # # Get transformed variables U
            # xa_particles = smf_transport_update(
            #     xf=(particles_forecast),                 # or X (after inflation) if you want it applied
            #     y=observation_y if observation_y.ndim == 2 else observation_y.unsqueeze(0).expand(B, -1),
            #     H=h_float,
            #     sigma_y=sigma_y,

            #     distMat=distMat,                      # your (d,d) state distance matrix (or None)
            #     offdiag_rad=float(0.0),              # your localization radius (float or None)

            #     p_rbf=float(p_rbf),                          # your p_rbf argument (1 or 2)
            #     nonId_radius=nonId_radius,            # OR pass K (see note below)
            #     M=M,                                  # your M argument (or None)

            #     # below are “extra knobs” that are NOT in your current function signature;
            #     # set them to safe defaults unless/until we port MATLAB faithfully:
            #     diag_order=float(2),
            #     lambda_=float(0.0),
            #     delta=float(1e-8),
            #     scalingRbf=float(2.0),
            # )
    else:
        X_firstK = X[:, :, :0]

    X_tail = X[:, :, K:] if K < d else X[:, :, :0]
    X_a = torch.cat([X_firstK, X_tail], dim=-1) if K < d else X_firstK

    # 6) order_all: keep linear (1). Lifting to higher order can be added later.
    # if K > 0:
    #     if p_rbf == 0:
    #         # linear case: A is defined above
    #         smf_map = SMFLinearKR(
    #             mu_z=mu_z, mu_x=mu_x, A=A, L=L,
    #             nonId_radius=K,
    #             meta={
    #                 "distMat_used": offdiag_rad is not None,
    #                 "offdiag_rad": offdiag_rad,
    #                 "rho": rho,
    #                 "order_all": order_all,
    #             },
    #         )
    #     else:
    #         # RBF case: smf_map already set to SMFSeparableKR above; do NOT touch it
    #         pass
    # else:
    #     B, m = X.shape[0], Yf.shape[-1]
    #     device, dtype = X.device, X.dtype
    #     smf_map = SMFLinearKR(
    #         mu_z=torch.zeros(B, m, device=device, dtype=dtype),          # (B,m)
    #         mu_x=torch.zeros(B, 0, device=device, dtype=dtype),          # (B,0)
    #         A=torch.zeros(B, m, 0, device=device, dtype=dtype),          # (B,m,0)
    #         L=torch.eye(0, device=device, dtype=dtype).unsqueeze(0).expand(B, 0, 0).contiguous(),
    #         nonId_radius=0,
    #         meta={"identity": True},
    #     )


    return X_a, smf_map


# ##############################################################################
# # Ensemble Kalman Filters (EnKF) (Analysis Step Only)
# ##############################################################################

def _enkf_pert_obs_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar or (B,)
    localization_matrix_Lxy=None, # (d_state, d_obs), broadcasts
    localization_matrix_Lyy=None,  # (d_obs, d_obs), broadcasts
    Gamma_Tildes=None,
    coords_state=None,
    loc_radius=None
):
    """ EnKF with Perturbed Observations - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    
    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) 
        d_obs = observation_y.shape[0]
    else:
        obs_y_eff = observation_y 
        d_obs = observation_y.shape[-1]
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    device = ensemble_f.device
    dtype = ensemble_f.dtype
    ensemble_y_f = observation_operator_ens(ensemble_f)

    Af, _ = center_ensemble(ensemble_f, rescale=False)
    AYf, _ = center_ensemble(ensemble_y_f, rescale=False)

    scaling_factor = 1.0 / (N_ensemble - 1) if N_ensemble > 1 else 1.0
    
    # Pxx = (Af.transpose(-2, -1) @ Af) * scaling_factor        
    # full_inds = torch.arange(0, d_state)
    # dist_xx = pairwise_distances(full_inds[:, None], full_inds[:, None], domain=(d_state,)).to(ensemble_f.device)
    # Dxx = dist2coeff(dist_xx, radius=loc_radius)
    # Pxx = (Af.transpose(-2, -1) @ Af) * scaling_factor * Dxx
    
    Pxy = (Af.transpose(-2, -1) @ AYf) * scaling_factor
    Pyy = (AYf.transpose(-2, -1) @ AYf) * scaling_factor

    if localization_matrix_Lxy is not None:
        Pxy = Pxy * localization_matrix_Lxy
    if localization_matrix_Lyy is not None:
        Pyy = Pyy * localization_matrix_Lyy
    

    I_obs = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0)         # (1, d_obs, d_obs)
    if Gamma_Tildes is None:
        # sigma_y can be scalar (tensor/float) or (B,)
        if isinstance(sigma_y, torch.Tensor):
            if sigma_y.ndim == 0:
                R_obs = (sigma_y**2) * I_obs.expand(batch_size, -1, -1)       # (B, d_obs, d_obs)
            elif sigma_y.ndim == 1 and sigma_y.shape[0] == batch_size:
                R_obs = (sigma_y.view(batch_size, 1, 1)**2) * I_obs           # (B, d_obs, d_obs)
            else:
                raise ValueError("sigma_y must be scalar or shape (B,).")
        else:
            # python float
            R_obs = (float(sigma_y)**2) * I_obs.expand(batch_size, -1, -1)    # (B, d_obs, d_obs)
    else:
        # Gamma_Tildes can be (d_obs,d_obs) or (B,d_obs,d_obs)
        if Gamma_Tildes.ndim == 2:
            if Gamma_Tildes.shape != (d_obs, d_obs):
                raise ValueError("Gamma_Tildes has wrong shape.")
            R_obs = Gamma_Tildes.unsqueeze(0).expand(batch_size, -1, -1)      # (B, d_obs, d_obs)
        elif Gamma_Tildes.ndim == 3:
            if Gamma_Tildes.shape[0] != batch_size or Gamma_Tildes.shape[1:] != (d_obs, d_obs):
                raise ValueError("Gamma_Tildes must be (B,d_obs,d_obs).")
            R_obs = Gamma_Tildes.to(device=device, dtype=dtype)
        else:
            raise ValueError("Gamma_Tildes must be 2D or 3D.")


    # if not args.access_to_noise:
        # for each sigma_y in the inputted sigma_y_batch, we want to sample 64 times from each sigma y
        # and compute the covariance matrix of basically h(x) where H(x) is the mean and Gamma is the covariance
    R_obs = R_obs.view(batch_size, d_obs, d_obs)
    innovation_cov = Pyy + R_obs
    # print(R_obs.shape, innovation_cov.shape)

    # --- MODIFICATION START ---
    # Add a small regularization term to innovation_cov to improve stability
    epsilon = 1e-6 # Regularization strength; adjust if necessary
    
    # Ensure reg_identity is correctly broadcastable for batched innovation_cov
    if innovation_cov.ndim == 3 and batch_size > 0 : # Batched
        reg_identity = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    elif innovation_cov.ndim == 2: # Non-batched (should not occur if inputs are batched)
         reg_identity = torch.eye(d_obs, device=device, dtype=dtype)
    else: # Handles batch_size = 0 or other unexpected dims for innovation_cov
        reg_identity = torch.eye(d_obs, device=device, dtype=dtype).unsqueeze(0)

    
    innovation_cov_reg = innovation_cov + epsilon * reg_identity
    # --- MODIFICATION END ---
    
    try:
        # K^T = solve(S_reg, Pxy^T) -> K = (solve(S_reg, Pxy^T))^T
        kalman_gain_T = torch.linalg.solve(innovation_cov_reg, Pxy.transpose(-2, -1))
        kalman_gain = kalman_gain_T.transpose(-2, -1)
    except torch.linalg.LinAlgError: # Catches errors like singularity if solve fails
        # Fallback to pseudo-inverse if solve fails even with regularization
        kalman_gain = Pxy @ torch.linalg.pinv(innovation_cov_reg)

    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim == 1 and sigma_y.shape[0] == batch_size:
        sigma_y_expanded = sigma_y.view(batch_size, 1, 1)
    else:
        sigma_y_expanded = sigma_y

    obs_perturbations = sigma_y_expanded * torch.randn(batch_size, N_ensemble, d_obs, device=device, dtype=dtype)
    perturbed_obs = obs_y_eff.unsqueeze(1) + obs_perturbations
    innovations = perturbed_obs - ensemble_y_f
    ensemble_a = ensemble_f + innovations @ kalman_gain.transpose(-2, -1)

    return ensemble_a, kalman_gain


def _esrf_analysis( # Ensemble Randomized Square Root Filter (ETKF variant) - Batched
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                 # scalar or (B,)
    Gamma_Tildes=None
):
    """ Ensemble Randomized Square Root Filter (ETKF) - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) # (1, d_obs)
    else:
        obs_y_eff = observation_y # (B, d_obs)
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")
    
    # N1: (N_ensemble - 1)
    N1_val = max(N_ensemble - 1.0, 1.0) # scalar

    # ensemble_y_f: (B, N_ensemble, d_obs)
    ensemble_y_f = observation_operator_ens(ensemble_f)

    # Af: (B, N_ensemble, d_state), mean_f: (B, 1, d_state)
    Af, mean_f = center_ensemble(ensemble_f, rescale=False)
    # AYf: (B, N_ensemble, d_obs), mean_yf: (B, 1, d_obs)
    AYf, mean_yf = center_ensemble(ensemble_y_f, rescale=False)

    # Prepare sigma_y for division, ensure it's (B,1,1) or scalar
    if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
        sigma_y_sq_inv = (1.0 / sigma_y**2).view(-1, 1, 1) # (B,1,1) or (1,1,1)
    else: # scalar
        sigma_y_sq_inv = 1.0 / (sigma_y**2)

    # C_tilde_sym: (B, N_ensemble, N_ensemble)
    # Original AYf @ AYf.T is (N_ens, N_y) @ (N_y, N_ens) -> (N_ens, N_ens)
    # Batched: (B, N_ens, N_y) @ (B, N_y, N_ens) -> (B, N_ens, N_ens)
    if Gamma_Tildes is not None:
        eps = 1e-6
        diag = torch.diagonal(Gamma_Tildes, dim1=-2, dim2=-1)
        jitter = eps * torch.diag_embed(diag)  # preserves the variance structure
        Gamma_pd = Gamma_Tildes + jitter
        # jitter = eps * torch.eye(d, device=Gamma_Tildes.device).unsqueeze(0)  # (1, d, d)
        # Gamma_pd = Gamma_Tildes + jitter  # (B, d, d)
        # Gamma_pd = Gamma_Tildes + eps * torch.mean(torch.diagonal(Gamma_Tildes, dim1=-2, dim2=-1), dim=-1, keepdim=True).unsqueeze(-1) * torch.eye(observation_y.shape[1], device=Gamma_Tildes.device).unsqueeze(0)
        L = torch.linalg.cholesky(Gamma_pd)  # (B, d, d)

        # Inverse of L
        L_inv = torch.linalg.pinv(L)  # (B, d, d)
        # U, S, Vh = torch.linalg.svd(Gamma_Tildes)  # (B, d, d), (B, d), (B, d, d)
        # S_clamped = torch.clamp(S, min=1e-6)
        # Gamma_inv_sqrt = (U @ torch.diag_embed(S_clamped**-0.5) @ Vh)

        # Then: Gamma^{-1/2} = L^{-T} = L_inv.transpose(-2, -1)
        Gamma_inv_sqrt = L_inv.transpose(-2, -1)  # (B, d, d)
        AYf = AYf @ Gamma_inv_sqrt
    if Gamma_Tildes is not None:
        # print((AYf @ AYf.transpose(-2, -1)).shape, Gamma_inv_sqrt.shape)
        C_tilde_sym = (AYf @ AYf.transpose(-2, -1)) + \
                       N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)
    else:
        C_tilde_sym = (AYf @ AYf.transpose(-2, -1)) * sigma_y_sq_inv + \
                N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)
    eig_vals, eig_vecs = robust_eigh(C_tilde_sym) # eig_vals (B,N), eig_vecs (B,N,N)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    # T_transform_matrix: (B, N_ensemble, N_ensemble)
    T_transform_matrix = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ \
                         eig_vecs.transpose(-2, -1) * torch.sqrt(torch.tensor(N1_val, device=device, dtype=dtype))
    # Pw_term: (B, N_ensemble, N_ensemble)
    Pw_term = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.transpose(-2, -1)

    # innovation_dy: (B, 1, d_obs)
    if Gamma_Tildes is not None:
        innovation_dy = (obs_y_eff.unsqueeze(1) - mean_yf) @ Gamma_inv_sqrt
        sigma_y_sq_inv = 1.0
    else:
        innovation_dy = (obs_y_eff.unsqueeze(1) - mean_yf)
    # w_gain_transpose: (B, 1, N_ensemble)
    w_gain_transpose = (innovation_dy @ AYf.transpose(-2, -1)) @ Pw_term * sigma_y_sq_inv
    # mean_a: (B, 1, d_state)
    mean_a = mean_f + w_gain_transpose @ Af
    # Af_updated: (B, N_ensemble, d_state)
    Af_updated = T_transform_matrix @ Af # T operates on rows of Af
    # ensemble_a: (B, N_ensemble, d_state)
    ensemble_a = mean_a + Af_updated

    return ensemble_a, None


def _letkf_core_etkf_update(
    local_E_f_mean,     # (B, N_x_local)
    local_A_f,          # (B, N_ens, N_x_local)
    eff_AY_f_anom,      # (B, N_ens, N_y_local)
    eff_d_f_innov,      # (B, N_y_local)
    N_ensemble,
    Gamma_inv_sqrt=None
):
    """ Core ETKF update for local patches (Batched), assuming R_eff = I. """
    device = local_A_f.device; dtype = local_A_f.dtype
    batch_size = local_A_f.shape[0]
    N1_val = max(N_ensemble - 1.0, 1.0) # scalar

    # Pa_tilde_inv_sqrt: (B, N_ensemble, N_ensemble)
    
    # if Gamma_inv_sqrt is not None:
    #     Pa_tilde_inv_sqrt = eff_AY_f_anom @ eff_AY_f_anom.transpose(-2, -1) @ Gamma_inv_sqrt + \
    #                     N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)
    # else:
    Pa_tilde_inv_sqrt = eff_AY_f_anom @ eff_AY_f_anom.transpose(-2, -1) + \
        N1_val * torch.eye(N_ensemble, device=device, dtype=dtype).unsqueeze(0)
    

    eig_vals, eig_vecs = robust_eigh(Pa_tilde_inv_sqrt)
    eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)

    # T_transform: (B, N_ensemble, N_ensemble)
    T_transform = eig_vecs @ torch.diag_embed(eig_vals_clamped**-0.5) @ \
                  eig_vecs.transpose(-2, -1) * torch.sqrt(torch.tensor(N1_val, device=device, dtype=dtype))
    # Pw: (B, N_ensemble, N_ensemble)
    Pw = eig_vecs @ torch.diag_embed(eig_vals_clamped**-1) @ eig_vecs.transpose(-2, -1)

    # w_gain_transpose: (B, 1, N_ensemble)
    w_gain_transpose = (eff_d_f_innov.unsqueeze(1) @ eff_AY_f_anom.transpose(-2, -1)) @ Pw

    # local_mean_a: (B, N_x_local)
    # (w_gain_transpose @ local_A_f) is (B, 1, N_x_local)
    mean_update_term = (w_gain_transpose @ local_A_f).squeeze(1) # (B, N_x_local)
    local_mean_a = local_E_f_mean + mean_update_term

    # local_A_a: (B, N_ensemble, N_x_local)
    local_A_a = T_transform @ local_A_f
    # print('Pw', Pw, 'local_A_f', local_A_f, 'local_A_a', local_A_a)

    return local_mean_a, local_A_a


def _letkf_analysis(
    ensemble_f,             # (B, N_ensemble, d_state)
    observation_y,          # (B, d_obs) or (d_obs,)
    observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                # scalar observation error standard deviation
    localization_radius,    # scalar
    coords_state,           # (d_state, D_coord_state)
    coords_obs,             # (d_obs, D_coord_obs)
    domain=None,     # (D_coord_state,) or (D_coord_obs,)
    Gamma_Tildes=None
):
    """ Local Ensemble Transform Kalman Filter (LETKF) - Analysis Step (Batched) """
    batch_size, N_ensemble, d_state = ensemble_f.shape
    device = ensemble_f.device; dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        obs_y_eff = observation_y.unsqueeze(0) # (1, d_obs)
        d_obs = observation_y.shape[0]
    else:
        obs_y_eff = observation_y # (B, d_obs)
        d_obs = observation_y.shape[-1]
        if obs_y_eff.shape[0] != batch_size and obs_y_eff.shape[0] != 1:
             raise ValueError("Batch size of observation_y must match ensemble_f or be 1.")

    # ensemble_y_f: (B, N_ensemble, d_obs)
    ensemble_y_f = observation_operator_ens(ensemble_f)
    # Af_global: (B, N_ensemble, d_state), mean_f_global: (B, 1, d_state)
    Af_global, mean_f_global = center_ensemble(ensemble_f, rescale=False)
    # AYf_global: (B, N_ensemble, d_obs), mean_yf_global: (B, 1, d_obs)
    AYf_global, mean_yf_global = center_ensemble(ensemble_y_f, rescale=False)

    # innovation_mean_global: (B, 1, d_obs)
    innovation_mean_global = obs_y_eff.unsqueeze(1) - mean_yf_global
    # print(innovation_mean_global.shape, 'first')
    
    B, N, d_obs = ensemble_y_f.shape
    # Transform observations and innovations by R^-1/2 (here R = sigma_y^2 * I)
    if Gamma_Tildes is not None:
        #insted of cholesky, use eigendecomposition to find inv square root
        eps = 1e-3  # or slightly larger if needed
        d = Gamma_Tildes.shape[-1]
        jitter = eps * torch.eye(d, device=Gamma_Tildes.device).unsqueeze(0)  # (1, d, d)
        Gamma_pd = Gamma_Tildes + jitter  # (B, d, d)
        L = torch.linalg.cholesky(Gamma_pd)  # (B, d, d)

        # Inverse of L
        L_inv = torch.linalg.pinv(L)  # (B, d, d)

        # Then: Gamma^{-1/2} = L^{-T} = L_inv.transpose(-2, -1)
        Gamma_inv_sqrt = L_inv.transpose(-2, -1)  # (B, d, d)
        # Gamma_sym = 0.5 * (Gamma_Tildes + Gamma_Tildes.transpose(-2, -1))
        # eigvals, eigvecs = torch.linalg.eigh(Gamma_sym)
        # eigvals_clamped = torch.clamp(eigvals, min=1e-9)
        # Gamma_inv_sqrt = eigvecs @ torch.diag_embed(eigvals_clamped.rsqrt()) @ eigvecs.transpose(-2, -1)

        AYf_global_transformed = AYf_global @ Gamma_inv_sqrt
        innovation_mean_global_transformed = innovation_mean_global @ Gamma_inv_sqrt
    else:
        sigma_y = torch.tensor(sigma_y, device=device, dtype=dtype)
        if sigma_y.ndim == 0:
            sigma_y_exp = sigma_y.view(1, 1, 1).expand(B, N, d_obs)
        elif sigma_y.ndim == 1:
            sigma_y_exp = sigma_y.view(B, 1, 1).expand(B, N, d_obs)
        else:
            raise ValueError("Unsupported shape for sigma_y")
        # print(sigma_y_exp.shape)
        AYf_global_transformed = AYf_global / sigma_y_exp         # (B, N_ensemble, d_obs)
        # print(AYf_global_transformed.shape)
        if sigma_y.ndim == 0:
            sigma_y_exp = sigma_y.view(1, 1, 1).expand(B, 1, d_obs)
        elif sigma_y.ndim == 1:
            sigma_y_exp = sigma_y.view(B, 1, 1).expand(B, 1, d_obs)
        else:
            raise ValueError("Unsupported shape for sigma_y")
        innovation_mean_global_transformed = innovation_mean_global / sigma_y_exp
        # print(innovation_mean_global_transformed.shape)
    # AYf_global_transformed = AYf_global / sigma_y         # (B, N_ensemble, d_obs)
    # innovation_mean_global_transformed = innovation_mean_global / sigma_y # (B, 1, d_obs)

    # Initialize analysis ensemble parts
    ensemble_a_mean_parts = torch.zeros_like(mean_f_global) # (B, 1, d_state)
    ensemble_a_anom_parts = torch.zeros_like(Af_global)   # (B, N_ensemble, d_state)

    # Loop over each state variable to update it locally
    for k_state_idx in range(d_state):
        # current_mean_f_k: (B, 1) mean of k-th state var for all batches
        current_mean_f_k = mean_f_global[:, :, k_state_idx]
        # current_Af_k: (B, N_ensemble, 1) anomalies of k-th state var
        current_Af_k = Af_global[:, :, k_state_idx].unsqueeze(-1)

        # --- Localization: This part is NOT batched over `batch_size` ---
        # --- It's computed once per k_state_idx as coords are shared ---
        # Coords for k-th state var: (1, D_coord)
        coord_k_state = coords_state[k_state_idx].unsqueeze(0)

        # Distances from k-th state variable to all observations: (1, d_obs)
        # Assumes pairwise_distances can handle (N,D) (M,D) -> (N,M) inputs
        # or a specific 2D version is used for these non-batched coordinates.
        dist_state_k_to_obs = pairwise_distances(
            coord_k_state, coords_obs, domain=domain
        ).squeeze(0) # -> (d_obs,)
        
        rho_k = dist2coeff(dist_state_k_to_obs, localization_radius) # (d_obs,)
        local_obs_indices = torch.where(rho_k > 1e-6)[0] # (N_y_local_k,)
        
        if len(local_obs_indices) == 0: # No observations influence this state variable
            ensemble_a_mean_parts[:, :, k_state_idx] = current_mean_f_k
            ensemble_a_anom_parts[:, :, k_state_idx] = current_Af_k.squeeze(-1)
            continue

        # Select local observations for this k_state_idx
        # These are now batched over `batch_size`
        # AYf_local_k_transformed: (B, N_ensemble, N_y_local_k)
        AYf_local_k = AYf_global_transformed[:, :, local_obs_indices]
        # innov_local_k_transformed: (B, 1, N_y_local_k)
        innov_local_k = innovation_mean_global_transformed[:, :, local_obs_indices]
        
        # Apply localization weights to observations (sqrt_rho acts on transformed obs anoms)
        rho_local_k_weights = rho_k[local_obs_indices] # (N_y_local_k,)
        # sqrt_rho_local_k broadcastable: (1, 1, N_y_local_k)
        sqrt_rho_local_k_bcast = torch.sqrt(rho_local_k_weights).view(1, 1, -1)

        # eff_AYf_k_anom: (B, N_ensemble, N_y_local_k)
        eff_AYf_k_anom = AYf_local_k * sqrt_rho_local_k_bcast
        # eff_innov_k: (B, 1, N_y_local_k) -> squeezed to (B, N_y_local_k)
        eff_innov_k = (innov_local_k * sqrt_rho_local_k_bcast).squeeze(1)

        # Core ETKF update for (k_state_idx, and all batches)
        # current_mean_f_k is (B,1), current_Af_k is (B, N_ens, 1)
        if Gamma_Tildes is not None:
            updated_mean_k, updated_A_k = _letkf_core_etkf_update(
                current_mean_f_k, current_Af_k,
                eff_AYf_k_anom, eff_innov_k, N_ensemble, Gamma_inv_sqrt=Gamma_inv_sqrt
            )
        else:
            updated_mean_k, updated_A_k = _letkf_core_etkf_update(
                current_mean_f_k, current_Af_k,
                eff_AYf_k_anom, eff_innov_k, N_ensemble
            )
        # updated_mean_k: (B,1), updated_A_k: (B, N_ensemble, 1)

        ensemble_a_mean_parts[:, :, k_state_idx] = updated_mean_k
        ensemble_a_anom_parts[:, :, k_state_idx] = updated_A_k.squeeze(-1)

    ensemble_a = ensemble_a_mean_parts + ensemble_a_anom_parts
    return ensemble_a, None

## iEnKS
# def _ienks_analysis(
#     ensemble_f,
#     observation_y,
#     observation_operator_ens,
#     sigma_y,
#     # --- Model specific args ---
#     model_propagator,
#     model_rhs,
#     model_dt,
#     # --- iEnKS hyperparameters ---
#     upd_a='Sqrt', # <-- MODIFIED: Added update method selector
#     Lag=1,
#     nIter=10,
#     wtol=1e-5,
#     steps_between_analyses=5
# ):
#     """
#     Function:
#         Implements a batched Iterative Ensemble Kalman Smoother (iEnKS) analysis step.
#     """
#     B, N, D_state = ensemble_f.shape
#     device = ensemble_f.device
#     dtype = ensemble_f.dtype

#     if observation_y.ndim == 1:
#         y = observation_y.unsqueeze(0)
#     else:
#         y = observation_y

#     N1 = N - 1
#     X0, x0 = center_ensemble(ensemble_f)

#     w = torch.zeros(B, N, 1, device=device, dtype=dtype)
#     T = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
#     Tinv = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
#     D_pert = None # For PertObs method

#     if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
#         R_inv_sqrt = (1.0 / sigma_y).view(-1, 1, 1)
#     else:
#         R_inv_sqrt = 1.0 / sigma_y

#     def propagate_ensemble_in_window(ens_in):
#         ens_flat = ens_in.view(B * N, D_state)
#         num_model_steps = Lag * steps_between_analyses
#         propagated_ens = ens_flat
#         for _ in range(num_model_steps):
#             propagated_ens = model_propagator(lambda x: model_rhs(x), propagated_ens, model_dt)
#         return propagated_ens.view(B, N, D_state)

#     for iteration in range(nIter):
#         E_iter = x0 + T @ X0 + (X0.transpose(-1, -2) @ w).transpose(-1, -2)
#         E_fwd = propagate_ensemble_in_window(E_iter)
#         Eo = observation_operator_ens(E_fwd)

#         Y, xo_obs = center_ensemble(Eo)
#         dy_eff = (y.unsqueeze(1) - xo_obs) * R_inv_sqrt
#         Y_eff = Y * R_inv_sqrt
#         za = float(N1)

#         Y_iter = Tinv @ Y_eff

#         # Unified Cow1 calculation for all methods
#         C_tilde = (Y_iter @ Y_iter.transpose(-2, -1)) + za * torch.eye(N, device=device, dtype=dtype)
#         eig_vals, U = torch.linalg.eigh(C_tilde)
#         eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)
#         Cow1 = U @ torch.diag_embed(1.0 / eig_vals_clamped) @ U.transpose(-2, -1)

#         # Gauss-Newton optimization for weights `w`
#         grad_term = Y_iter @ dy_eff.transpose(-2, -1)
#         grad = grad_term - za * w
#         dw = Cow1 @ grad

#         # MODIFIED: Update transform matrices T and Tinv based on the chosen method
#         if "Sqrt" in upd_a:
#             eig_vals_sqrt = torch.sqrt(eig_vals_clamped)
#             T = U @ torch.diag_embed(1.0 / eig_vals_sqrt) @ U.transpose(-2,-1) * math.sqrt(N1)
#             Tinv = U @ torch.diag_embed(eig_vals_sqrt) @ U.transpose(-2,-1) / math.sqrt(N1)
#         elif "PertObs" in upd_a:
#             if iteration == 0:
#                 _D_pert = torch.randn_like(Y_eff)
#                 D_pert = _D_pert - _D_pert.mean(dim=1, keepdim=True)
#             gradT = -(Y_eff + D_pert) @ Y_iter.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T)
#             T = T + gradT @ Cow1
#             Tinv = torch.linalg.inv(T + 1)
#         elif "Order1" in upd_a:
#             gradT = -0.5 * Y_eff @ Y_iter.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T)
#             T = T + gradT @ Cow1
#             Tinv = torch.linalg.inv(T)
#         else:
#             raise NotImplementedError(f"Update type '{upd_a}' not implemented.")

#         w_new = w + dw
#         if ((w_new - w).norm(p=2, dim=1)**2 / N).mean() < wtol:
#             w = w_new
#             break
#         w = w_new

#     final_delta_mean = (X0.transpose(-2, -1) @ w).transpose(-1, -2)
#     final_X_smoothed = T @ X0
#     E_smoothed_at_start = x0 + final_delta_mean + final_X_smoothed

#     return E_smoothed_at_start, None

def _ienks_analysis(
    ensemble_f,
    observation_y,
    observation_operator_ens,
    sigma_y,
    sigma_v,
    # --- Localization and Model Args ---
    localization_radius,
    coords_state=None,
    coords_obs=None,
    domain=None,
    model_propagator=None,
    model_rhs=None,
    model_dt=None,
    # --- iEnKS hyperparameters ---
    upd_a='Sqrt',
    Lag=1,
    nIter=10,
    wtol=1e-5,
    steps_between_analyses=5,
    Gamma_Tildes=None
):
    # print('-'*80)
    B, N, D_state = ensemble_f.shape
    if Gamma_Tildes is not None:
        B = Gamma_Tildes.shape[0]
    device = ensemble_f.device
    dtype = ensemble_f.dtype

    if observation_y.ndim == 1:
        y = observation_y.unsqueeze(0)
    else:
        y = observation_y
    R_inv_sqrt = None
    if Gamma_Tildes is not None:
        eps = 1e-3  # or slightly larger if needed
        d = Gamma_Tildes.shape[-1]
        jitter = eps * torch.eye(d, device=Gamma_Tildes.device).unsqueeze(0)  # (1, d, d)
        G = Gamma_Tildes + jitter  # (B, d, d)
        L = torch.linalg.cholesky(G.view(B, observation_y.shape[1], observation_y.shape[1]))  # (B, d, d)
        # Inverse of L
        L_inv = torch.linalg.inv(L)  # (B, d, d)
        # Then: Gamma^{-1/2} = L^{-T} = L_inv.transpose(-2, -1)
        R_inv_sqrt = L_inv.transpose(-2, -1)  # (B, d, d)
    else:
        if isinstance(sigma_y, torch.Tensor) and sigma_y.ndim > 0:
            R_inv_sqrt = (1.0 / sigma_y).view(1, 1, -1)
        else:
            R_inv_sqrt = 1.0 / sigma_y
    # check_tensor("Gamma_Tildes", Gamma_Tildes)
    # check_tensor("G after jitter", G)
    # check_tensor("L (Cholesky)", L)
    # check_tensor("R_inv_sqrt", R_inv_sqrt)

    N1 = N - 1
    X0_global, x0_global = center_ensemble(ensemble_f)
    def propagate_ensemble_in_window(ens_in):
        ens_flat = ens_in.view(B * N, D_state)
        num_model_steps = Lag * steps_between_analyses
        propagated_ens = ens_flat
        for _ in range(num_model_steps):
            propagated_ens = model_propagator(model_rhs, propagated_ens, 0, model_dt)
        propagated_ens += torch.randn_like(propagated_ens, device=propagated_ens.device) * sigma_v
        return propagated_ens.view(B, N, D_state)
    
    # ============================================================================
    # --- Conditional Path: Global or Local Analysis ---
    # ============================================================================

    if localization_radius is None:
        w = torch.zeros(B, N, 1, device=device, dtype=dtype)
        T = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        D_pert = None

        for iteration in range(nIter):
            E_iter = x0_global + T @ X0_global + (X0_global.transpose(-1, -2) @ w).transpose(-1, -2)
            E_fwd = propagate_ensemble_in_window(E_iter)
            Eo = observation_operator_ens(E_fwd)

            Y, xo_obs = center_ensemble(Eo)
            # dy_eff = (y.unsqueeze(1) - xo_obs) * R_inv_sqrt
            # Y_eff = Y * R_inv_sqrt
            if Gamma_Tildes is not None:   # full Gamma case (B,d_obs,d_obs)
                dy_eff = (y.unsqueeze(1) - xo_obs) @ R_inv_sqrt.transpose(-2,-1)
                Y_eff  = Y @ R_inv_sqrt.transpose(-2,-1)
            else:  # diag case
                dy_eff = (y.unsqueeze(1) - xo_obs) * R_inv_sqrt
                Y_eff  = Y * R_inv_sqrt
            
            za = float(N1)
            Tinv = torch.linalg.inv(T)
            Y_iter = Tinv @ Y_eff
            
            C_tilde = (Y_iter @ Y_iter.transpose(-2, -1)) + za * torch.eye(N, device=device).unsqueeze(0)

            eig_vals, U = robust_eigh(C_tilde)
            eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)
            Cow1 = U @ torch.diag_embed(1.0 / eig_vals_clamped) @ U.transpose(-2, -1)
            
            grad = (Y_iter @ dy_eff.transpose(-2, -1)) - za * w
            dw = Cow1 @ grad

            if "Sqrt" in upd_a:
                T = U @ torch.diag_embed(1.0 / torch.sqrt(eig_vals_clamped)) @ U.transpose(-2,-1) * math.sqrt(N1)
            elif "PertObs" in upd_a:
                if iteration == 0:
                    _D_pert = torch.randn_like(Y_eff)
                    D_pert = _D_pert - _D_pert.mean(dim=1, keepdim=True)
                gradT = -(Y_eff + D_pert) @ Y_iter.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T)
                T = T + gradT @ Cow1
            elif "Order1" in upd_a:
                gradT = -0.5 * Y_eff @ Y_iter.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T)
                T = T + gradT @ Cow1
            
            w_new = w + dw
            if ((w_new - w).norm(p=2, dim=1)**2 / N).mean() < wtol: w = w_new; break
            w = w_new
        
        final_delta_mean = (X0_global.transpose(-2, -1) @ w).transpose(-1, -2)
        final_X_smoothed = T @ X0_global
        # check_tensor("E_iter", E_iter)
        # check_tensor("E_fwd", E_fwd)
        # check_tensor("Eo", Eo)
        # check_tensor("Y (obs anomalies)", Y)
        # check_tensor("xo_obs", xo_obs)
        # check_tensor("dy_eff", dy_eff)
        # check_tensor("Y_eff", Y_eff)
        # check_tensor("Y_iter", Y_iter)
        # check_tensor("C_tilde", C_tilde)
        # check_tensor("eig_vals", eig_vals)
        # check_tensor("grad", grad)
        # check_tensor("dw", dw)
        # check_tensor("T", T)
    else:
        if coords_state is None or coords_obs is None:
            raise ValueError("coords_state and coords_obs must be provided for localization.")
            
        final_delta_mean = torch.zeros_like(x0_global)
        final_X_smoothed = torch.zeros_like(X0_global)

        for k_state_idx in range(D_state):
            coord_k_state = coords_state[k_state_idx].unsqueeze(0)
            dist_k_to_obs = pairwise_distances(coord_k_state, coords_obs, domain=domain).squeeze(0)
            rho_k = dist2coeff(dist_k_to_obs, localization_radius)
            local_obs_indices = torch.where(rho_k > 1e-6)[0]

            X0_k = X0_global[:, :, k_state_idx].unsqueeze(-1)

            if len(local_obs_indices) == 0:
                final_delta_mean[:, :, k_state_idx] = 0.0
                final_X_smoothed[:, :, k_state_idx] = X0_k.squeeze(-1)
                continue
            
            y_local = y[:, local_obs_indices]
            rho_local_k = rho_k[local_obs_indices]
            sqrt_rho_bcast = torch.sqrt(rho_local_k).view(1, 1, -1)
            
            w_k = torch.zeros(B, N, 1, device=device, dtype=dtype)
            T_k = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            D_pert_local = None

            for iteration in range(nIter):
                E_iter = x0_global + T_k @ X0_global + (X0_global.transpose(-2, -1) @ w_k).transpose(-1, -2)
                E_fwd = propagate_ensemble_in_window(E_iter)
                Eo = observation_operator_ens(E_fwd)

                Eo_local = Eo[:, :, local_obs_indices]
                Y_local, xo_obs_local = center_ensemble(Eo_local)
                # dy_local_eff = (y_local.unsqueeze(1) - xo_obs_local) * R_inv_sqrt * sqrt_rho_bcast
                # Y_local_eff = Y_local * R_inv_sqrt * sqrt_rho_bcast
                if Gamma_Tildes is not None:   # full Gamma case
                    # Extract block corresponding to local obs
                    R_inv_sqrt_local = R_inv_sqrt[:, local_obs_indices][:, :, local_obs_indices]  # (B, n_local, n_local)
                    # R_inv_sqrt_local = torch.stack([
                    #     R_inv_sqrt[b][local_obs_indices][:, local_obs_indices] for b in range(B)
                    # ], dim=0)
                    dy_local_eff = (y_local.unsqueeze(1) - xo_obs_local) @ R_inv_sqrt_local.transpose(-2,-1)
                    Y_local_eff  = Y_local @ R_inv_sqrt_local.transpose(-2,-1)
                    # then apply localization
                    dy_local_eff = dy_local_eff * sqrt_rho_bcast
                    Y_local_eff  = Y_local_eff * sqrt_rho_bcast
                else:  # diag case
                    dy_local_eff = (y_local.unsqueeze(1) - xo_obs_local) * R_inv_sqrt * sqrt_rho_bcast
                    Y_local_eff  = Y_local * R_inv_sqrt * sqrt_rho_bcast
                
                za = float(N1)
                # Tinv_k = torch.linalg.inv(T_k)
                # Y_iter_local = Tinv_k @ Y_local_eff
                Y_iter_local = torch.linalg.solve(T_k, Y_local_eff)

                C_tilde = (Y_iter_local @ Y_iter_local.transpose(-2, -1)) + za*torch.eye(N, device=device)
                eig_vals, U = robust_eigh(C_tilde)
                eig_vals_clamped = torch.clamp(eig_vals, min=1e-9)
                Cow1 = U @ torch.diag_embed(1.0 / eig_vals_clamped) @ U.transpose(-2, -1)

                grad = (Y_iter_local @ dy_local_eff.transpose(-2, -1)) - za * w_k
                dw_k = Cow1 @ grad
                
                if "Sqrt" in upd_a:
                    T_k = U @ torch.diag_embed(1.0 / torch.sqrt(eig_vals_clamped)) @ U.transpose(-2,-1) * math.sqrt(N1)
                elif "PertObs" in upd_a:
                    if iteration == 0:
                        _D_pert_local = torch.randn_like(Y_local_eff)
                        D_pert_local = _D_pert_local - _D_pert_local.mean(dim=1, keepdim=True)
                    gradT_k = -(Y_local_eff + D_pert_local) @ Y_iter_local.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T_k)
                    T_k = T_k + gradT_k @ Cow1
                elif "Order1" in upd_a:
                    gradT_k = -0.5 * Y_local_eff @ Y_iter_local.transpose(-2, -1) + N1 * (torch.eye(N, device=device) - T_k)
                    T_k = T_k + gradT_k @ Cow1
                
                w_k_new = w_k + dw_k
                if ((w_k_new - w_k).norm(p=2, dim=1)**2 / N).mean() < wtol: w_k = w_k_new; break
                w_k = w_k_new
            
            delta_mean_k = (X0_k.transpose(-2, -1) @ w_k).transpose(-1, -2)
            X_smoothed_k = T_k @ X0_k
            final_delta_mean[:, :, k_state_idx] = delta_mean_k.squeeze(-1)
            final_X_smoothed[:, :, k_state_idx] = X_smoothed_k.squeeze(-1)
            # check_tensor("rho_k", rho_k)
            # check_tensor("y_local", y_local)
            # check_tensor("x0_global", x0_global)
            # check_tensor("T_k", T_k)
            # check_tensor("X0_global", X0_global)
            # check_tensor("w_k", w_k)
            # check_tensor("E_iter", E_iter)
            # check_tensor("E_fwd", E_fwd)
            # check_tensor("Eo_local", Eo_local)
            # check_tensor("Y_local", Y_local)
            # check_tensor("dy_local_eff", dy_local_eff)
            # check_tensor("Y_local_eff", Y_local_eff)
            # check_tensor("C_tilde local", C_tilde)
            # check_tensor("eig_vals local", eig_vals)
            # check_tensor("grad local", grad)
            # check_tensor("dw_k", dw_k)
            # check_tensor("T_k", T_k)
            # check_tensor("w_k", w_k)

    E_smoothed_at_start = x0_global + final_delta_mean + final_X_smoothed
    #check if e_smoothed_at_start is nan
    #run comprehensive diagnostics and check a lot of the intermediate variables
    
        
    # check_tensor("final_delta_mean", final_delta_mean)
    # check_tensor("final_X_smoothed", final_X_smoothed)
    return E_smoothed_at_start, None


# EnKF analysis
# def ensemble_kalman_filter_analysis(
#     ensemble_f,             # (B, N_ensemble, d_state)
#     observation_y,          # (B, d_obs) or (d_obs,) or None
#     observation_operator_ens, # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
#     sigma_y,                # scalar or (B,)
#     method="EnKF-PertObs",
#     inflation_factor=1.0,   # scalar
#     # For EnKF-PertObs
#     localization_matrix_Lxy=None, # (d_state, d_obs)
#     localization_matrix_Lyy=None, # (d_obs, d_obs)
#     # For LETKF
#     localization_radius_letkf=None, # scalar
#     coords_state_letkf=None,        # (d_state, D_coord)
#     coords_obs_letkf=None,          # (d_obs, D_coord)
#     domain_letkf=None,      # (D_coord,)
#     # For iEnKS
#     ienks_lag=1,
#     ienks_niter=10,
#     ienks_wtol=1e-5,
#     model_args=None # Dict for model propagator info needed by iEnKS
# ):
#     """ Main dispatcher for ensemble Kalman filter analysis (Batched) """
#     kalman_gain_or_transform = None
#     ensemble_a_raw = None

#     if observation_y is None: # No observation, forecast is analysis
#         ensemble_a_raw = ensemble_f
#     elif method == "EnKF-PertObs":
#         ensemble_a_raw, kalman_gain_or_transform = _enkf_pert_obs_analysis(
#             ensemble_f, observation_y, observation_operator_ens, sigma_y,
#             localization_matrix_Lxy, localization_matrix_Lyy
#         )
#     elif method == "ESRF": # ETKF variant
#         ensemble_a_raw, kalman_gain_or_transform = _esrf_analysis(
#             ensemble_f, observation_y, observation_operator_ens, sigma_y
#         )
#     elif method == "LETKF":
#         if localization_radius_letkf is None or \
#            coords_state_letkf is None or \
#            coords_obs_letkf is None:
#             raise ValueError("LETKF requires localization_radius, coords_state, and coords_obs.")
#         ensemble_a_raw, kalman_gain_or_transform = _letkf_analysis(
#             ensemble_f, observation_y, observation_operator_ens, sigma_y,
#             localization_radius_letkf, coords_state_letkf,
#             coords_obs_letkf, domain_letkf
#         )
#     elif method.startswith("iEnKS-"):
#         if model_args is None:
#             raise ValueError("iEnKS methods require 'model_args' dictionary.")
        
#         # Extract update type from method name, e.g., "iEnKS-Sqrt" -> "Sqrt"
#         try:
#             update_type = method.split('-', 1)[1]
#         except IndexError:
#             raise ValueError(f"Invalid iEnKS method format: {method}. Expected 'iEnKS-UpdateType'.")

#         ensemble_a_raw, kalman_gain_or_transform = _ienks_analysis(
#             ensemble_f, observation_y, observation_operator_ens, sigma_y,
#             model_propagator=model_args['propagator'],
#             model_rhs=model_args['rhs'],
#             model_dt=model_args['dt'],
#             steps_between_analyses=model_args['steps_between_analyses'],
#             upd_a=update_type, # Pass the extracted update type
#             Lag=ienks_lag,
#             nIter=ienks_niter,
#             wtol=ienks_wtol,
#         )
#     else:
#         raise ValueError(f"Unknown EnKF method: {method}")

#     # Apply inflation to the raw analysis ensemble
#     ensemble_analysis = apply_inflation(ensemble_a_raw, inflation_factor)

#     return ensemble_analysis, kalman_gain_or_transform

def ensemble_kalman_filter_analysis(
    ensemble_f,                 # (B, N_ensemble, d_state)
    observation_y,              # (B, d_obs) or (d_obs,) or None
    observation_operator_ens,   # (B, N_ensemble, d_state) -> (B, N_ensemble, d_obs)
    sigma_y,                    # scalar or (B,)
    sigma_v,                    # scalar or (B,)
    method="EnKF-PertObs",
    inflation_factor=1.0,       # scalar
    # --- Parameters for Covariance Localization (e.g., for EnKF-PertObs) ---
    localization_matrix_Lxy=None, # (d_state, d_obs)
    localization_matrix_Lyy=None, # (d_obs, d_obs)
    # --- SHARED Parameters for Observation Space Localization (LETKF, iEnKS) ---
    localization_radius=None,   # scalar or None
    coords_state=None,          # (d_state, D_coord)
    coords_obs=None,            # (d_obs, D_coord)
    localization_domain=None,   # (D_coord,)
    # --- Parameters for iEnKS ---
    ienks_lag=1,
    ienks_niter=10,
    ienks_wtol=1e-5,
    model_args=None,            # Dict for model propagator info needed by iEnKS
    Gamma_Tilde=None,           # None, (B, d_obs, d_obs), or per-valid-slice
    invalid_trajs=None,
):
    """
    Main dispatcher for ensemble Kalman filter analysis (Batched).
    Supports passing optional Gamma_Tilde and optional invalid_trajs.
    Ensures we never index into None and only slice when needed.
    """
    # Prepare validity mask
    if invalid_trajs is None:
        valid_mask = None
    else:
        valid_mask = ~invalid_trajs

    # Filter inputs for valid trajectories only if invalid_trajs provided
    if valid_mask is not None:
        ensemble_f_valid = ensemble_f[valid_mask]
        observation_y_valid = observation_y[valid_mask] if observation_y is not None else None
        Gamma_Tilde_valid = Gamma_Tilde[valid_mask] if (Gamma_Tilde is not None) else None
    else:
        ensemble_f_valid = ensemble_f
        observation_y_valid = observation_y
        Gamma_Tilde_valid = Gamma_Tilde

    kalman_gain_or_transform = None
    ensemble_a_valid = None

    # If no observation, analysis = forecast
    if observation_y_valid is None:
        ensemble_a_valid = ensemble_f_valid

    elif method == "EnKF-PertObs":
        ensemble_a_valid, kalman_gain_or_transform = _enkf_pert_obs_analysis(
            ensemble_f_valid, observation_y_valid, observation_operator_ens, sigma_y,
            localization_matrix_Lxy, localization_matrix_Lyy, Gamma_Tilde_valid,
            coords_state, localization_radius
        )
    elif method == "ESRF":  # ETKF variant
        ensemble_a_valid, kalman_gain_or_transform = _esrf_analysis(
            ensemble_f_valid, observation_y_valid, observation_operator_ens, sigma_y, Gamma_Tilde_valid
        )

    elif method == "LETKF":
        if (localization_radius is None) or (coords_state is None) or (coords_obs is None):
            raise ValueError("LETKF requires localization_radius, coords_state, and coords_obs.")
        ensemble_a_valid, kalman_gain_or_transform = _letkf_analysis(
            ensemble_f_valid, observation_y_valid, observation_operator_ens, sigma_y,
            localization_radius, coords_state, coords_obs, localization_domain, Gamma_Tilde_valid
        )

    elif method.startswith("iEnKS"):
        if model_args is None:
            raise ValueError("iEnKS methods require 'model_args' dictionary.")
        # Extract update type from method name, e.g., "iEnKS-Sqrt" -> "Sqrt"
        try:
            update_type = method.split('-', 1)[1]
        except IndexError:
            raise ValueError(f"Invalid iEnKS method format: {method}. Expected 'iEnKS-UpdateType'.")

        ensemble_a_valid, kalman_gain_or_transform = _ienks_analysis(
            # DA args
            ensemble_f_valid, observation_y_valid, observation_operator_ens, sigma_y, sigma_v,
            # Localization args
            localization_radius=localization_radius,
            coords_state=coords_state,
            coords_obs=coords_obs,
            domain=localization_domain,
            # Model args
            model_propagator=model_args['propagator'],
            model_rhs=model_args['rhs'],
            model_dt=model_args['dt'],
            steps_between_analyses=model_args['steps_between_analyses'],
            # iEnKS hyperparams
            upd_a=update_type,
            Lag=ienks_lag,
            nIter=ienks_niter,
            wtol=ienks_wtol,
            Gamma_Tildes=Gamma_Tilde_valid
        )
    else:
        raise ValueError(f"Unknown EnKF method: {method}")

    # Reconstitute to full batch if we filtered
    if valid_mask is not None:
        B = ensemble_f.shape[0]
        _ensemble_a_raw = torch.full_like(ensemble_f, float('nan'))
        _ensemble_a_raw[valid_mask] = ensemble_a_valid
        ensemble_a_raw = _ensemble_a_raw
    else:
        ensemble_a_raw = ensemble_a_valid

    # Apply inflation
    ensemble_analysis = apply_inflation(ensemble_a_raw, inflation_factor)
    return ensemble_analysis, kalman_gain_or_transform



# =======================================================================
# Stochastic Map Filter extensions (appended)
# =======================================================================
# def stochastic_map_filter_analysis(
#     ensemble_f,
#     observation_y,
#     sigma_y,
#     obs_indices,
#     options,
# ):
#     """
#     Apply the stochastic-map filter analysis step to a batch of forecast ensembles.
#     Parameters
#     ----------
#     ensemble_f : torch.Tensor
#         Forecast ensembles of shape (B, N, d_state).
#     observation_y : torch.Tensor
#         Observations for the current assimilation step, shape (B, d_obs).
#     sigma_y : float or torch.Tensor
#         Observation noise standard deviation(s).
#     obs_indices : Iterable[int]
#         Indices of the observed state variables.
#     options : dict
#         Dictionary configuring the stochastic map filter (distMat, order_all, etc.).
#     """
#     import torch
#     from types import SimpleNamespace
#     try:
#         from stochastic_maps_py.methods import StochasticMapFilter
#     except ImportError as exc:  # pragma: no cover - dependency missing
#         raise RuntimeError("stochastic_maps_py package is required for SMF support.") from exc
#     ensemble_f = ensemble_f
#     batch_size, ensemble_size, state_dim = ensemble_f.shape
#     dist_matrix = torch.as_tensor(options.get("distMat"), dtype=torch.int64)
#     order_all = int(options.get("order_all", 2))
#     nonid_radius = int(options.get("nonId_radius", state_dim))
#     offdiag_radius = int(options.get("offdiag_rad", state_dim))
#     rho = float(options.get("rho", 0.0))
#     if torch.is_tensor(obs_indices):
#         obs_idx_list = [int(idx) for idx in obs_indices.tolist()]
#     else:
#         obs_idx_list = [int(idx) for idx in obs_indices]
#     if isinstance(sigma_y, torch.Tensor):
#         sigma_vec = sigma_y.detach().cpu().to(torch.double).reshape(-1)
#     else:
#         sigma_vec = torch.full((batch_size,), float(sigma_y), dtype=torch.double)
#     analyses = []
#     for b in range(batch_size):
#         sigma_val = float(sigma_vec[min(b, sigma_vec.numel() - 1)])
#         def sample_likelihood(x, noise_std=sigma_val):
#             return x + noise_std * torch.randn_like(x)
#         sm_options = {
#             "M": options.get("M", ensemble_size),
#             "distMat": dist_matrix,
#             "order_all": order_all,
#             "nonId_radius": nonid_radius,
#             "offdiag_rad": offdiag_radius,
#             "rho": rho,
#         }
#         model = SimpleNamespace(
#             d=state_dim,
#             data_idx=obs_idx_list,
#             data_indices=obs_idx_list,  # new line
#             sample_likelihood=sample_likelihood,
#         )
#         sm_filter = StochasticMapFilter(model, sm_options)
#         forecast = ensemble_f[b].detach().cpu().double()
#         obs = observation_y[b].detach().cpu().double()
#         analysis = sm_filter.sample_posterior(forecast, obs)
#         analyses.append(analysis.to(dtype=ensemble_f.dtype, device=ensemble_f.device))
#     return torch.stack(analyses, dim=0)

# ##############################################################################
# # Lorenz 96 and RK4 for Testing
# ##############################################################################
def lorenz96_rhs(x, F=8):
    """
    Calculates the RHS of the Lorenz 96 equations.
    x can be a 1D tensor (D_state,) or a 2D tensor (batch_size, D_state).
    F is a scalar forcing term.
    """
    D = x.shape[-1] # Works for both 1D and 2D x

    # Efficiently calculate indices for all D components
    # Indices are relative to the current component 'k'
    # x_k-2, x_k-1, x_k, x_k+1
    # For dxdt[k] = (x[k+1] - x[k-2]) * x[k-1] - x[k] + F

    # Create rolled versions of x for vectorized computation
    # x_m2 means x[(i-2+D)%D] for each i
    # x_m1 means x[(i-1+D)%D] for each i
    # x_p1 means x[(i+1)%D] for each i
    x_m2 = torch.roll(x, shifts=2, dims=-1)
    x_m1 = torch.roll(x, shifts=1, dims=-1)
    x_p1 = torch.roll(x, shifts=-1, dims=-1)

    dxdt = (x_p1 - x_m2) * x_m1 - x + F
    return dxdt

# def lorenz63_rhs(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
#     """
#     # Function: Calculates the RHS of the Lorenz 63 equations.
#     # ---
#     # Input:
#     #   x (torch.Tensor): State tensor of shape (batch_size, 3) or (3,).
#     #   sigma, rho, beta (float): Lorenz 63 parameters.
#     # ---
#     # Output:
#     #   torch.Tensor: The derivatives (dx/dt, dy/dt, dz/dt) with the same shape as x.
#     """
#     # Unpack state variables
#     x_val = x[..., 0]
#     y_val = x[..., 1]
#     z_val = x[..., 2]

#     # Lorenz 63 equations
#     dxdt = sigma * (y_val - x_val)
#     dydt = x_val * (rho - z_val) - y_val
#     dzdt = x_val * y_val - beta * z_val

#     # Stack the results back into a tensor of the same shape as the input
#     return torch.stack([dxdt, dydt, dzdt], dim=-1)

def rk4_step(rhs_func, x, t, dt):
    """
    Performs one RK4 step.
    x can be (D_state,) or (batch_size, D_state).
    rhs_func is compatible with batched x.
    """
    k1 = rhs_func(x)
    k2 = rhs_func(x + 0.5 * dt * k1)
    k3 = rhs_func(x + 0.5 * dt * k2)
    k4 = rhs_func(x + dt * k3)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

class Lorenz63:
    """
    Function:
        Implements the Lorenz 63 model with an RK4 stepper.
    """
    def __init__(self, sigma=10.0, rho=28.0, beta=8./3.):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _rhs(self, x):
        """
        Function:
            Computes the right-hand side of the Lorenz 63 equations.
        Input:
            x (torch.Tensor): State vector(s) of shape [N, 3] or [3]. It can handle batches, e.g., [B*N, 3].
        Output:
            torch.Tensor: The derivative dx/dt for each state vector.
        """
        is_1d = x.ndim == 1
        if is_1d:
            x = x.unsqueeze(0)
        
        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = self.sigma * (x[:, 1] - x[:, 0])
        dxdt[:, 1] = x[:, 0] * (self.rho - x[:, 2]) - x[:, 1]
        dxdt[:, 2] = x[:, 0] * x[:, 1] - self.beta * x[:, 2]
        
        return dxdt.squeeze(0) if is_1d else dxdt

    def step(self, rhs_func, x, t, dt):
        """
        Function:
            Advances the model state by one time step using RK4.
        Input:
            rhs_func (callable): The right-hand side function.
            x (torch.Tensor): Current state vector(s).
            dt (float): Time step size.
        Output:
            torch.Tensor: State vector(s) at the next time step.
        """
        k1 = rhs_func(x)
        k2 = rhs_func(x + dt * k1 / 2)
        k3 = rhs_func(x + dt * k2 / 2)
        k4 = rhs_func(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
# ##############################################################################
# # Main Test Script
# ##############################################################################
if __name__ == '__main__':
    import argparse
    # --- 0. Command-Line Argument Parser
    # Setup argument parser to select DA methods to run.
    parser = argparse.ArgumentParser(description="Run Data Assimilation Benchmark")
    parser.add_argument(
        '--methods', 
        nargs='+', 
        # default=['iEnKS-PertObs', 'iEnKS-Sqrt', 'iEnKS-Order1'], # Default methods to run
        default=['iEnKS-PertObs'], # Default methods to run
        choices=['BPF', 'EnKF-PO', 'ESRF', 'iEnKS-PertObs', 'iEnKS-Sqrt', 'iEnKS-Order1'],
        help='A list of DA methods to run.'
    )
    args = parser.parse_args()
    methods_to_run = args.methods
    print(f"Running selected methods: {methods_to_run}")
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # -- 1. Experiment Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # --- System and DA Parameters ---
    lorenz_model = Lorenz63()
    dt = 0.03
    obs_every = 5
    total_obs = 500
    n_steps = total_obs * obs_every
    batch_size = 32
    
    params = {
        'N': 10,
        'Lag': 1,
        'nIter': 10,
        'wtol': 1e-5,
        'infl': 1.08,
        'obs_noise_std': 1.0,
        'model_noise_std': 1e-2
    }
    
    # --- Observation Setup ---
    obs_inds = [0] # Observe only the 'x' variable
    obs_operator = lambda x: x[..., obs_inds]
    l63_rhs_func = lambda x: lorenz_model._rhs(x)
    rk4_stepper = lambda rhs, x, t, dt: lorenz_model.step(rhs, x, t, dt)

    print("Generating true state with model spin-up...")
    x_spinup = torch.randn(batch_size, 3, device=device, dtype=torch.float32)
    num_spinup_steps = 2000
    for _ in range(num_spinup_steps):
        x_spinup = rk4_stepper(l63_rhs_func, x_spinup, None, dt)
    print("Spin-up complete.")

    print("Generating full true trajectory batch...")
    x_true = torch.zeros((batch_size, n_steps + 1, 3), device=device, dtype=torch.float32)
    x_true[:, 0] = x_spinup
    for i in range(n_steps):
        x_true[:, i+1] = rk4_stepper(l63_rhs_func, x_true[:, i], None, dt)

    obs_time_indices = torch.arange(0, n_steps + 1, obs_every)
    xx_true_obs = x_true[:, obs_time_indices]
    
    true_obs_vals = xx_true_obs[:, :, obs_inds]
    yy = true_obs_vals + torch.randn_like(true_obs_vals) * params['obs_noise_std']
    
    # -- 3. Initialize Ensembles for All Methods
    print("Initializing ensembles for selected methods...")
    # All methods start from the same initial state
    initial_true_state_b = xx_true_obs[:, 0, :].unsqueeze(1)
    noise = torch.randn(batch_size, params['N'], 3, device=device) * 1.0
    initial_ensemble = initial_true_state_b + noise

    # Initialize dictionaries for results and ensembles based on selected methods
    results_rmse = {method: [] for method in methods_to_run}
    analysis_times = {method: [] for method in methods_to_run}
    ensemble_states = {method: initial_ensemble.clone() for method in methods_to_run}

    # -- 4. Run Assimilation Loop
    print(f"Running {total_obs} analysis cycles...")
    for ko in tqdm(range(total_obs), desc="DA Benchmark"):
        
        # --- A: Standard Filter Implementations ---
        # 1. Forecast step for filters
        filter_forecasts = {}
        for name in methods_to_run:
            # Skip iEnKS here, as its forecast logic is handled differently
            if name.startswith('iEnKS'):
                continue
            
            ens = ensemble_states[name]
            ens_flat = ens.view(-1, 3) # Shape: [batch_size * N, 3]
            for _ in range(obs_every):
                ens_flat = rk4_stepper(l63_rhs_func, ens_flat, None, dt)
            forecast = ens_flat.view(batch_size, params['N'], 3) 
            forecast += torch.randn_like(forecast) * params['model_noise_std']
            filter_forecasts[name] = forecast

        # 2. Analysis step for filters
        y_current = yy[:, ko, :]

        if 'BPF' in methods_to_run:
            start_time = time.perf_counter()
            analysis_bpf = bootstrap_particle_filter_analysis(
                filter_forecasts["BPF"], y_current, obs_operator, params['obs_noise_std'], resampling_method="systematic"
            )
            analysis_times["BPF"].append(time.perf_counter() - start_time)
            results_rmse["BPF"].append(torch.sqrt(torch.mean((analysis_bpf.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1)).cpu())
            ensemble_states["BPF"] = analysis_bpf

        common_enkf_args = {
            "observation_y": y_current,
            "observation_operator_ens": obs_operator,
            "sigma_y": params['obs_noise_std'],
            "sigma_v": params['model_noise_std'],
            "inflation_factor": params['infl']
        }
        
        if 'EnKF-PO' in methods_to_run:
            start_time = time.perf_counter()
            analysis_enkf_po, _ = ensemble_kalman_filter_analysis(filter_forecasts["EnKF-PO"], method="EnKF-PertObs", **common_enkf_args)
            analysis_times["EnKF-PO"].append(time.perf_counter() - start_time)
            results_rmse["EnKF-PO"].append(torch.sqrt(torch.mean((analysis_enkf_po.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1)).cpu())
            ensemble_states["EnKF-PO"] = analysis_enkf_po

        if 'ESRF' in methods_to_run:
            start_time = time.perf_counter()
            analysis_esrf, _ = ensemble_kalman_filter_analysis(filter_forecasts["ESRF"], method="ESRF", **common_enkf_args)
            analysis_times["ESRF"].append(time.perf_counter() - start_time)
            results_rmse["ESRF"].append(torch.sqrt(torch.mean((analysis_esrf.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1)).cpu())
            ensemble_states["ESRF"] = analysis_esrf

        # --- B: iEnKS Implementation (Smoother Logic) ---
        ienks_methods = [m for m in methods_to_run if m.startswith('iEnKS-')]

        if ienks_methods:
            # This setup is common for all iEnKS methods
            k_start = max(0, ko - params['Lag'])
            
            model_args_ienks = {
                "propagator": rk4_stepper,
                "rhs": l63_rhs_func,
                "dt": dt,
                "steps_between_analyses": obs_every,
            }

            # Loop through each specific iEnKS method (e.g., iEnKS-Sqrt, iEnKS-PertObs)
            for method_name in ienks_methods:
                # 1. Analysis step for the specific iEnKS method
                start_time = time.perf_counter()
                E_smoothed_at_start, _ = ensemble_kalman_filter_analysis(
                    ensemble_f=ensemble_states[method_name], # Use the forecast for this method
                    observation_y=y_current,
                    observation_operator_ens=obs_operator,
                    sigma_y=params['obs_noise_std'],
                    sigma_v=params['model_noise_std'],
                    # sigma_v=0,
                    method=method_name, # Pass the specific method name
                    inflation_factor=params['infl'],
                    ienks_lag=(ko - k_start),
                    ienks_niter=params['nIter'],
                    ienks_wtol=params['wtol'],
                    model_args=model_args_ienks
                )
                analysis_times[method_name].append(time.perf_counter() - start_time)

                # 2. Propagate smoothed state to current time `ko` for RMSE calculation
                E_analysis_at_ko = E_smoothed_at_start.clone()
                num_steps_to_propagate = (ko - k_start) * obs_every
                if num_steps_to_propagate > 0:
                    E_flat = E_analysis_at_ko.view(-1, 3)
                    for _ in range(num_steps_to_propagate):
                        E_flat = rk4_stepper(l63_rhs_func, E_flat, None, dt)
                    E_analysis_at_ko = E_flat.view(batch_size, params['N'], 3)
                
                # Store results for the current method
                rmse = torch.sqrt(torch.mean((E_analysis_at_ko.mean(dim=1) - xx_true_obs[:, ko, :])**2, dim=-1))
                results_rmse[method_name].append(rmse.cpu())

                # 3. Create the forecast for the *next* cycle's window start
                E_flat_next = E_smoothed_at_start.view(-1, 3)
                for _ in range(obs_every):
                    E_flat_next = rk4_stepper(l63_rhs_func, E_flat_next, None, dt)
                forecast_next = E_flat_next.view(batch_size, params['N'], 3)
                
                # Add model noise for the next forecast
                forecast_next += torch.randn_like(forecast_next) * params['model_noise_std']
                
                # Update the state for the current method
                ensemble_states[method_name] = forecast_next


    # -- 5. Final Results
    print("\n--- Final Average Metrics (Stable Period) ---")
    for method_name in methods_to_run:
        rmse_tensor = torch.stack(results_rmse[method_name])
        time_tensor = torch.tensor(analysis_times[method_name])
        print("RMSE tensor shape:", rmse_tensor.shape)
        stable_start_idx = 0
        rmse_mean = rmse_tensor[stable_start_idx:, :].mean().item() 
        avg_time = time_tensor[stable_start_idx:].mean().item() 
        
        print(f"{method_name:<8s}: Avg. Time: {avg_time:.6f} s | Mean RMSE (stable): {rmse_mean:.4f}")