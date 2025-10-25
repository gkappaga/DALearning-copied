from __future__ import annotations

from typing import Dict

import torch

from stochastic_maps_py.data_assimilation.seq_assimilation import SequentialAssimilationResult
from stochastic_maps_py.models import DataAssimilationModel


def post_process(model: DataAssimilationModel, result: SequentialAssimilationResult, options: Dict) -> Dict:
    start_idx = int(round(model.t0 / (model.dt * model.dt_iter)))
    idx_range = slice(start_idx, start_idx + model.J)
    truth = model.xt[idx_range]

    burn_in = getattr(model, "T_BurnIn", 0)

    mean_est = result.mean[burn_in:]
    variance_est = result.variance[burn_in:]
    quantile_est = result.quantiles[burn_in:]
    truth_eval = truth[burn_in:]

    diff = mean_est - truth_eval
    rmse = torch.sqrt(torch.sum(diff ** 2, dim=1) / model.d)

    spread = torch.sqrt(torch.sum(variance_est, dim=1) / model.d)

    lower = quantile_est[:, 0]
    upper = quantile_est[:, 1]
    coverage = ((truth_eval >= lower) & (truth_eval <= upper)).float().mean(dim=1)

    summary = {
        "RMSE": rmse,
        "RMSE_mean": rmse.mean(),
        "RMSE_median": rmse.median(),
        "RMSE_std": rmse.std(unbiased=False),
        "spread": spread,
        "spread_mean": spread.mean(),
        "spread_median": spread.median(),
        "spread_std": spread.std(unbiased=False),
        "coverage_prob": coverage,
        "covprob_mean": coverage.mean(),
        "covprob_median": coverage.median(),
        "covprob_std": coverage.std(unbiased=False),
    }

    for key, value in options.items():
        summary[key] = value

    return summary


__all__ = ["post_process"]
