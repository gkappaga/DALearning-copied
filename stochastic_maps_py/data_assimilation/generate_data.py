from __future__ import annotations

from typing import Tuple

import torch

from stochastic_maps_py.models import DataAssimilationModel


def generate_data(model: DataAssimilationModel, num_steps: int) -> DataAssimilationModel:
    """Generate state and observation trajectories matching the MATLAB helper."""

    state = model.x0.clone().to(torch.double)
    time = float(model.t0)
    dt = model.dt
    dt_iter = model.dt_iter

    states = []
    observations = []
    times = []

    obs_matrix = model.obs_matrix.to(torch.double)

    for _ in range(num_steps):
        for _ in range(dt_iter):
            state = model.for_op(state, time, dt)
            time += dt
        state = state + model.sigma_x * torch.randn_like(state)
        obs_state = torch.matmul(obs_matrix, state.T).T
        observation = model.sample_likelihood(obs_state.view(-1))
        states.append(state.clone())
        observations.append(torch.as_tensor(observation, dtype=torch.double).flatten())
        times.append(time)

    xt = torch.stack(states)
    yt = torch.stack(observations)
    tt = torch.tensor(times, dtype=torch.double)

    return model.clone_with(xt=xt, yt=yt, tt=tt)


__all__ = ["generate_data"]
