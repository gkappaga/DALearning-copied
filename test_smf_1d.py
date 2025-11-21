import torch
from benchmark_analysis_v2 import *

# If your code lives in a module, do:
# from my_smf_module import stochastic_map_filter_analysis

def test_smf_1d_linear_gaussian():
    torch.manual_seed(0)

    # ---- Forecast ensemble: x ~ N(0,1) ----
    B, N, d = 1, 100_000, 1        # big N to kill Monte Carlo noise
    prior_mean = 0.0
    prior_std = 1.0
    X_forecast = prior_mean + prior_std * torch.randn(B, N, d)

    # ---- Observation model y = x + eps ----
    # H is identity: y = x, so H = [1] with shape (m,d) = (1,1)
    H = torch.ones(1, 1)

    sigma_y = 0.1                  # obs noise std
    y_star_value = 0.3
    # observation_y must be (B, m)
    observation_y = torch.tensor([[y_star_value]], dtype=torch.float32)

    # ---- Run SMF analysis ----
    X_a, smf_map = stochastic_map_filter_analysis(
        particles_forecast=X_forecast,
        observation_y=observation_y,
        observation_operator=H,
        sigma_y=sigma_y,
        M=None,
        distMat=None,
        order_all=1,
        nonId_radius=None,
        offdiag_rad=None,
        rho=0.0,
        jitter=1e-6,
    )


    # ---- Sample posterior mean/variance from X_a ----
    # X_a shape: (B, N, d); here B=1,d=1
    mean_a = X_a.mean(dim=1)           # (B,d)
    var_a  = X_a.var(dim=1, unbiased=True)

    sample_mean = mean_a.item()
    sample_var  = var_a.item()

    print("Sample posterior mean:", sample_mean)
    print("Sample posterior var :", sample_var)

    # ---- Analytic Kalman posterior for comparison ----
    prior_var = 1.0
    obs_var   = sigma_y ** 2

    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    K = prior_var / (prior_var + obs_var)
    post_mean = K * y_star_value

    print("Analytic posterior mean:", post_mean)
    print("Analytic posterior var :", post_var)

    # Rough sanity assertions (you can tighten/loosen these)
    assert abs(sample_mean - post_mean) < 5e-3, "Mean too far from analytic"
    assert abs(sample_var  - post_var)  < 5e-3, "Var too far from analytic"

if __name__ == "__main__":
    test_smf_1d_linear_gaussian()
