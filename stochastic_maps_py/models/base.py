from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import torch


StatePropagator = Callable[[torch.Tensor, float, float], torch.Tensor]
LikelihoodSampler = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DataAssimilationModel:
    d: int
    dt: float
    dt_iter: int
    sigma_x: float
    for_op: StatePropagator
    sample_likelihood: LikelihoodSampler
    data_indices: Sequence[int]
    obs_matrix: torch.Tensor

    x0: torch.Tensor
    t0: float = 0.0
    J: int = 0

    m0: Optional[torch.Tensor] = None
    C0: Optional[torch.Tensor] = None

    xt: Optional[torch.Tensor] = None
    yt: Optional[torch.Tensor] = None
    tt: Optional[torch.Tensor] = None

    T_SpinUp: Optional[int] = None
    T_Steps: Optional[int] = None
    T_BurnIn: int = 0

    def clone_with(self, **updates) -> "DataAssimilationModel":
        return replace(self, **updates)


__all__ = ["DataAssimilationModel", "StatePropagator", "LikelihoodSampler"]
