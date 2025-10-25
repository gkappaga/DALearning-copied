from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from stochastic_maps_py.methods.transport_map_component import TransportMapComponent


@dataclass
class TransportMap:
    order: Sequence[Sequence[int]]
    options: dict

    def __post_init__(self) -> None:
        self.order = [list(int(v) for v in component) for component in self.order]
        self.dimension = len(self.order)
        self.components: List[TransportMapComponent] = []
        for idx, comp_order in enumerate(self.order, start=1):
            component = TransportMapComponent(idx, comp_order, self.options)
            self.components.append(component)

    def optimize(self, samples: torch.Tensor, non_identity_components: Iterable[int]) -> "TransportMap":
        for idx in non_identity_components:
            self.components[idx - 1].optimize(samples[:, :idx])
        return self

    def eval_map(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.double)
        N, d = samples.shape
        outputs = torch.zeros((N, d), dtype=torch.double)
        for idx in range(d):
            outputs[:, idx] = self.components[idx].evaluate(samples[:, : idx + 1])
        return outputs

    def eval_inv_map(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.double)
        N, d = samples.shape
        outputs = torch.zeros((N, d), dtype=torch.double)
        for idx in range(d):
            prev = outputs[:, :idx] if idx > 0 else torch.zeros((N, 0), dtype=torch.double)
            outputs[:, idx] = self.components[idx].inverse(prev, samples[:, idx])
        return outputs

    def grad_x(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.to(torch.double)
        N, d = samples.shape
        gradients = torch.zeros((N, d, d), dtype=torch.double)
        for idx in range(d):
            gradients[:, idx, : idx + 1] = self.components[idx].grad_x(samples[:, : idx + 1])
        return gradients


__all__ = ["TransportMap"]
