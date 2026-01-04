# smf_fullobs_diagnostics.py
# ------------------------------------------------------------
# Theory-level checks for your linear SMF implementation.
# Verifies:
#   1) K == d when m == d (full observation)
#   2) Covariance identities: Σxz≈Σxx, Σzz≈Σxx+R (R=I for σ_y=1)
#   3) A ≈ (Σxx+R)^{-1} Σxx and Σx|z matches analytic Schur complement
#   4) Round-trip identity: inverse ∘ forward ≈ identity when y* = y^i
#   5) Matrix-H vs callable-H equivalence (optional)
# ------------------------------------------------------------

import torch
import math
import importlib
import os
import sys

# --- adjust if your file/module name differs ---
IMPORT_PATH = "benchmark_analysis_v2"  # your python file without .py
mod = importlib.import_module(IMPORT_PATH)

# Pull needed functions
_smf_apply_H           = getattr(mod, "_smf_apply_H")
_smf_fit_linear_KR     = getattr(mod, "_smf_fit_linear_KR")
_smf_forward_linear_KR = getattr(mod, "_smf_forward_linear_KR")
_smf_inverse_linear_KR = getattr(mod, "_smf_inverse_linear_KR")
stochastic_map_filter_analysis = getattr(mod, "stochastic_map_filter_analysis")

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Config (feel free to tweak)
# --------------------------
B = 2            # batches
N = 2000         # ensemble size (bigger => cleaner cov checks)
d = 3            # state dimension (use 3 for L63-like)
sigma_y = 1.0    # STD (R = I for tests)
seed = 123

# --------------------------
# Helpers
# --------------------------
def rel_err(a, b, eps=1e-12):
    na = a.norm()
    nb = b.norm()
    d  = (a - b).norm()
    denom = max(float(nb), eps)
    return float(d / denom)

@torch.no_grad()
def run_fullobs_cov_checks(X):
    """
    X: (B,N,d) forecast ensemble
    Uses H = I (matrix path) and σ=1 to build Zs exactly like your SMF,
    then reproduces the fit stats to verify identities.
    """
    B, N, d = X.shape
    H = torch.eye(d, device=X.device, dtype=X.dtype)  # full observation, matrix path
    Yf = _smf_apply_H(X, H)                           # (B,N,d) == X
    Zs = Yf + torch.randn_like(Yf) * sigma_y          # σ=1 => R = I

    # Centered covariances (match your code)
    mu_z = Zs.mean(1) ; Zc = Zs - mu_z.unsqueeze(1)
    mu_x = X.mean(1)  ; Xc = X  - mu_x.unsqueeze(1)

    Szz = (Zc.transpose(1,2) @ Zc) / (N - 1)          # (B,d,d)
    Sxz = (Xc.transpose(1,2) @ Zc) / (N - 1)          # (B,d,d)
    Sxx = (Xc.transpose(1,2) @ Xc) / (N - 1)          # (B,d,d)

    # Theory identities for H=I and R=I:
    R = torch.eye(d, device=X.device, dtype=X.dtype).unsqueeze(0)  # (B,d,d) (broadcast ok)

    e_xz = rel_err(Sxz, Sxx)            # Σxz ≈ Σxx
    e_zz = rel_err(Szz, Sxx + R)        # Σzz ≈ Σxx + R

    # Compute your A and Σx|z via your fit function
    mu_z_fit, mu_x_fit, A, L = _smf_fit_linear_KR(Zs, X, distMat=None, offdiag_rad=None, jitter=0.0)

    # Analytic A*: (Σxx + R)^{-1} Σxx
    # Use Cholesky solves (no inverses)
    L_zz = torch.linalg.cholesky(Sxx + R)
    T    = torch.linalg.solve_triangular(L_zz, Sxx, upper=False)
    A_star = torch.linalg.solve_triangular(L_zz.transpose(1,2), T, upper=True)  # (B,d,d)

    e_A  = rel_err(A, A_star)

    # Σx|z (yours) vs analytic
    Sx_given_z = Sxx - Sxz @ A
    Sx_star    = Sxx - Sxx @ A_star
    e_Sc = rel_err(Sx_given_z, Sx_star)

    return {
        "rel||Σxz-Σxx||": e_xz,
        "rel||Σzz-(Σxx+R)||": e_zz,
        "rel||A-A*||": e_A,
        "rel||Σx|z-Σx|z*||": e_Sc,
        "A_shape": tuple(A.shape),
        "L_shape": tuple(L.shape),
    }

@torch.no_grad()
def round_trip_identity_check(X):
    """
    Draw Zs ~ p(z|x), set y* = z_i for each particle, and verify:
        inverse(y*, forward(y*, x)) ≈ x
    """
    B, N, d = X.shape
    H = torch.eye(d, device=X.device, dtype=X.dtype)
    Yf = _smf_apply_H(X, H)                           # (B,N,d) == X
    Zs = Yf + torch.randn_like(Yf) * sigma_y          # σ=1

    # Fit map
    mu_z, mu_x, A, L = _smf_fit_linear_KR(Zs, X, distMat=None, offdiag_rad=None, jitter=0.0)

    # Forward then inverse with y* = z_i (per-particle)
    # forward expects z: (B,N,m), x:(B,N,d)
    U = _smf_forward_linear_KR(mu_z, mu_x, A, L, Zs, X)            # (B,N,d)
    X_back = _smf_inverse_linear_KR(mu_z, mu_x, A, L, Zs, U)       # (B,N,d)

    num = (X_back - X).norm()
    den = X.norm().clamp_min(1e-12)
    rel = float(num / den)
    return rel

@torch.no_grad()
def run_driver_compare_H_paths():
    """
    Compare using matrix H (I_d) vs callable H that returns the same.
    Ensures _smf_apply_H callable branch behaves identically.
    """
    torch.manual_seed(seed + 99)
    X = torch.randn(B, N, d, device=device)

    # matrix path
    H_mat = torch.eye(d, device=device, dtype=X.dtype)
    Y_mat = _smf_apply_H(X, H_mat)

    # callable path that should mimic matrix path; try batched signature
    def H_fun_batched(xBNd):  # xBNd: (B,N,d)
        return xBNd  # identity

    Y_fun = _smf_apply_H(X, H_fun_batched)

    e = rel_err(Y_fun, Y_mat)
    return e

@torch.no_grad()
def assert_full_obs_K_equals_d():
    torch.manual_seed(seed + 7)
    X = torch.randn(B, N, d, device=device)
    y = torch.zeros(B, d, device=device)  # not used by the test

    # Use matrix H=I
    H = torch.eye(d, device=device, dtype=X.dtype)
    # Force full observation (m=d); set nonId_radius=None to intend K=d
    X_a, smf_map = stochastic_map_filter_analysis(
        particles_forecast=X,
        observation_y=y,
        observation_operator=H,
        sigma_y=sigma_y,
        M=N,
        distMat=None,
        order_all=1,
        nonId_radius=None,      # <-- if this is changed elsewhere in your code, the assert below will fail
        offdiag_rad=None,
        rho=0.0,
        jitter=0.0,
    )
    # If your function exposes nonId_radius on the returned map:
    if hasattr(smf_map, "nonId_radius"):
        K = smf_map.nonId_radius
        assert K == d, f"Expected K=d={d} under full observation, but got K={K}. Check nonId_radius."

def main():
    torch.manual_seed(seed)
    X = torch.randn(B, N, d, device=device)

    print("=== Full-observation theory checks (H = I_d, σ=1) ===")
    stats = run_fullobs_cov_checks(X)
    for k, v in stats.items():
        print(f"{k:>28s}: {v}")

    print("\n=== Round-trip identity (y* = z_i) ===")
    rel = round_trip_identity_check(X)
    print(f"rel || inverse(forward(x,z), z) - x || / ||x|| = {rel:.3e}  (should be ~1e-6 to 1e-8)")

    print("\n=== Matrix-H vs Callable-H path equivalence ===")
    eH = run_driver_compare_H_paths()
    print(f"rel || H_matrix(X) - H_callable(X) || / ||H_matrix(X)|| = {eH:.3e}  (should be ~0)")

    print("\n=== Assert K == d when m == d ===")
    try:
        assert_full_obs_K_equals_d()
        print("K == d check passed.")
    except AssertionError as ae:
        print(f"[FAIL] {ae}")

if __name__ == "__main__":
    main()
