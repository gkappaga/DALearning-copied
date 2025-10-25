from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch

from stochastic_maps_py.models import DataAssimilationModel


AnalysisCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SequentialAssimilationResult:
    posterior: torch.Tensor
    time: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    quantiles: torch.Tensor


def seq_assimilation(model: DataAssimilationModel, analysis: AnalysisCallable) -> SequentialAssimilationResult:
    dt = model.dt
    dt_iter = model.dt_iter
    sigma_x = model.sigma_x

    xf = model.x0.clone().to(torch.double)
    tf = float(model.t0)

    num_steps = model.J
    d = model.d

    mean_t = torch.zeros((num_steps, d), dtype=torch.double)
    var_t = torch.zeros((num_steps, d), dtype=torch.double)
    cov_t = torch.zeros((num_steps, d, d), dtype=torch.double)
    quantiles = torch.zeros((num_steps, 2, d), dtype=torch.double)
    time_t = torch.zeros(num_steps, dtype=torch.double)

    start_idx = int(round(model.t0 / (dt * dt_iter)))
    obs_sequence = model.yt[start_idx:start_idx + num_steps]

    posterior = xf.clone()

    for step, observation in enumerate(obs_sequence):
        for _ in range(dt_iter):
            xf = model.for_op(xf, tf, dt)
            tf += dt
        xf = xf + sigma_x * torch.randn_like(xf)

        xp = analysis(xf, observation)

        time_t[step] = tf
        mean_t[step] = xp.mean(dim=0)
        var_t[step] = xp.var(dim=0, unbiased=False)
        cov_t[step] = torch.cov(xp.T)
        quantiles[step] = torch.quantile(xp, torch.tensor([0.025, 0.975], dtype=torch.double), dim=0)

        posterior = xp
        xf = xp

    return SequentialAssimilationResult(
        posterior=posterior,
        time=time_t,
        mean=mean_t,
        variance=var_t,
        covariance=cov_t,
        quantiles=quantiles,
    )


__all__ = ["seq_assimilation", "SequentialAssimilationResult"]
