from __future__ import annotations

from typing import Callable

import torch

from stochastic_maps_py.data_assimilation.seq_assimilation import seq_assimilation
from stochastic_maps_py.models import DataAssimilationModel


AnalysisFactory = Callable[[DataAssimilationModel], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


def spin_up(model: DataAssimilationModel, ensemble_size: int, factory: AnalysisFactory) -> DataAssimilationModel:
    if model.m0 is None or model.C0 is None:
        raise ValueError("Spin-up requires prior mean `m0` and covariance `C0`.")
    chol = torch.linalg.cholesky(model.C0.to(torch.double))
    samples = model.m0.to(torch.double).unsqueeze(1) + chol @ torch.randn(
        model.d, ensemble_size, dtype=torch.double
    )
    spin_model = model.clone_with(x0=samples.T, t0=0.0, J=model.T_SpinUp)
    algorithm = factory(spin_model)
    result = seq_assimilation(spin_model, algorithm)
    updated_model = model.clone_with(
        x0=result.posterior,
        t0=result.time[-1].item(),
        J=model.T_Steps,
    )
    return updated_model


__all__ = ["spin_up"]
