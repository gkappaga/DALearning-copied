from __future__ import annotations

from typing import Tuple

import torch

from config.dataset_info import DATASET_INFO
from stochastic_maps_py.data_assimilation import generate_data
from stochastic_maps_py.methods import StochasticMapFilter
from stochastic_maps_py.models import DataAssimilationModel
from stochastic_maps_py.tools import lorenz_distance_matrix
from utils import L63, rk4


def build_lorenz63_problem(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    sigma_x: float = 1e-2,
    sigma_y: float = 2.0,
    dt: float = 0.05,
    dt_iter: int = 2,
    seed: int = 10,
    ensemble_size: int = 200,
    T_spin_up: int = 2000,
    T_steps: int = 2000,
) -> Tuple[DataAssimilationModel, dict]:
    torch.manual_seed(seed)

    params = DATASET_INFO.get("lorenz63", {})
    d = params.get("dim", 3)
    obs_indices = params.get("obs_inds", torch.arange(d))
    obs_indices = obs_indices.to(torch.long)
    H = torch.eye(d, dtype=torch.float64)[obs_indices]

    def dynamics(state: torch.Tensor, time: float, step_size: float) -> torch.Tensor:
        def rhs(t, x):
            return L63.forward(t, x, sig=sigma, rho=rho, beta=beta)

        return rk4(rhs, state, time, step_size)

    def sample_likelihood(state: torch.Tensor) -> torch.Tensor:
        return state + sigma_y * torch.randn_like(state)

    m0 = torch.zeros(d, dtype=torch.float64)
    C0 = torch.eye(d, dtype=torch.float64)
    x0_data = m0.unsqueeze(0)

    base_model = DataAssimilationModel(
        d=d,
        dt=dt,
        dt_iter=dt_iter,
        sigma_x=sigma_x,
        for_op=dynamics,
        sample_likelihood=sample_likelihood,
        data_indices=obs_indices.tolist(),
        obs_matrix=H,
        x0=x0_data,
        m0=m0,
        C0=C0,
        T_SpinUp=T_spin_up,
        T_Steps=T_steps,
        T_BurnIn=0,
        J=T_spin_up + T_steps,
    )

    base_model = generate_data(base_model, base_model.J)

    start_idx = T_spin_up - 1
    state_start = base_model.xt[start_idx]
    ensemble = state_start + 0.1 * torch.randn(ensemble_size, d, dtype=torch.float64)

    model = base_model.clone_with(
        x0=ensemble,
        t0=base_model.tt[start_idx].item(),
        J=T_steps,
    )

    options = {
        "M": ensemble_size,
        "distMat": lorenz_distance_matrix(d),
        "order_all": 2,
        "nonId_radius": d,
        "offdiag_rad": d,
        "rho": 0.1,
    }

    return model, options


def sample_run() -> StochasticMapFilter:
    model, options = build_lorenz63_problem()
    return StochasticMapFilter(model, options)


__all__ = ["build_lorenz63_problem", "sample_run"]
