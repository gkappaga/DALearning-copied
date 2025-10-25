from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from stochastic_maps_py.methods.transport_map import TransportMap
from stochastic_maps_py.models import DataAssimilationModel


@dataclass
class StochasticMapFilter:
    model: DataAssimilationModel
    options: Dict[str, float]

    def __post_init__(self) -> None:
        self.options = self._default_options(dict(self.options))
        self.order, self.non_identity_components = self._setup_transport_orders()
        self.transport_map = TransportMap(self.order, self.options)
        all_components = set(range(1, len(self.order) + 1))
        identity_components = all_components.difference(self.non_identity_components)
        for idx in identity_components:
            self.transport_map.components[idx - 1].set_identity_component()

    def sample_posterior(self, forecast_samples: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        data_indices = self.model.data_indices
        obs_vector = torch.as_tensor(observation, dtype=torch.double).flatten()
        analysis = forecast_samples.clone().to(torch.double)
        for idx, obs_value in zip(data_indices, obs_vector):
            analysis = self.assimilate_scalar_observation(analysis, int(idx), float(obs_value))
        return analysis

    def assimilate_scalar_observation(self, forecast_samples: torch.Tensor, data_index: int, obs_value: float) -> torch.Tensor:
        dist_mat = self._distance_matrix()
        permutation = torch.argsort(dist_mat[:, data_index])
        reordered = forecast_samples[:, permutation]

        inflated = self.inflate(reordered)
        Yi = self._sample_likelihood(inflated[:, 0])
        inputs = torch.column_stack([Yi, inflated])
        self.transport_map.optimize(inputs, self.non_identity_components)

        Xi = reordered[:, 0]
        Yi_post = self._sample_likelihood(Xi)
        mapped = self.evaluate(torch.column_stack([Yi_post, reordered]), obs_value)

        result = torch.zeros_like(reordered)
        result[:, permutation] = mapped
        return result

    def inflate(self, samples: torch.Tensor) -> torch.Tensor:
        mean = samples.mean(dim=0, keepdim=True)
        delta = torch.sqrt(torch.tensor(1.0 + self.options["rho"], dtype=torch.double)) * (samples - mean)
        return delta + mean

    def evaluate(self, observations_states: torch.Tensor, obs_value: float) -> torch.Tensor:
        observations_states = observations_states.to(torch.double)
        Eta = self.transport_map.eval_map(observations_states)
        Eta[:, 0] = obs_value
        posterior = self.transport_map.eval_inv_map(Eta)
        return posterior[:, 1:]

    def gradient(self, observations_states: torch.Tensor, obs_value: float) -> torch.Tensor:
        observations_states = observations_states.to(torch.double)
        gradients = self.transport_map.grad_x(observations_states)
        mapped_states = self.evaluate(observations_states, obs_value)
        obs_column = torch.full((observations_states.size(0),), obs_value, dtype=torch.double)
        gradients_obs = self.transport_map.grad_x(torch.column_stack([obs_column, mapped_states]))
        inv_grad = gradients_obs[:, 1:, 1:]
        result = torch.zeros((observations_states.size(0), gradients.shape[1] - 1, gradients.shape[2]), dtype=torch.double)
        for n in range(observations_states.size(0)):
            result[n] = torch.linalg.solve(inv_grad[n], gradients[n, 1:])
        return result

    def _setup_transport_orders(self) -> (List[List[int]], List[int]):
        d_model = self.model.d
        order: List[List[int]] = [[] for _ in range(d_model + 1)]
        dist_mat = self._distance_matrix()

        offdiag_order = int(self.options["offdiag_order"])
        offdiag_radius = int(self.options["offdiag_rad"])
        dist_to_order = torch.zeros(d_model, dtype=torch.int64)
        dist_to_order[:offdiag_radius] = offdiag_order

        permutation = torch.argsort(dist_mat[:, 0])
        non_identity: List[int] = []

        for i in range(d_model):
            node_i = permutation[i].item()
            dist_i_1 = dist_mat[0, node_i].item()
            if dist_i_1 <= self.options["nonId_radius"]:
                orders_i = torch.zeros(i, dtype=torch.int64)
                for j in range(i):
                    node_j = permutation[j].item()
                    dist_i_j = int(dist_mat[node_i, node_j].item())
                    orders_i[j] = dist_to_order[dist_i_j]
                diag_order = (
                    int(self.options["diag_order_obs"]) if i == 0 else int(self.options["diag_order_unobs"])
                )
                orders_i = torch.cat([torch.tensor([diag_order], dtype=torch.int64), orders_i])
                if (i == 0 and self.options["locLik"] == 1) or (self.options["locLik"] == 0):
                    data_order = int(self.options["data_order"])
                else:
                    data_order = 0
                component_order = torch.cat([torch.tensor([data_order], dtype=torch.int64), orders_i])
                order[i + 1] = component_order.tolist()
                non_identity.append(i + 1)

        for idx in range(1, d_model + 2):
            if not order[idx - 1]:
                identity = [0] * (idx - 1) + [1]
                order[idx - 1] = identity

        return order, non_identity

    def _distance_matrix(self) -> torch.Tensor:
        return torch.as_tensor(self.options["distMat"], dtype=torch.int64)

    @staticmethod
    def _default_options(options: Dict[str, float]) -> Dict[str, float]:
        order_all = options.get("order_all", 1)
        defaults = {
            "scalingWidths": 2.0,
            "lambda": 0.0,
            "delta": 1e-8,
            "npoints_interp": 2000,
            "kappa": 4.0,
            "data_order": order_all,
            "offdiag_order": order_all,
            "diag_order_obs": order_all,
            "diag_order_unobs": 1,
            "locLik": 1,
            "nonId_radius": 1,
            "offdiag_rad": 1,
            "rho": 0.0,
        }
        for key, value in defaults.items():
            options.setdefault(key, value)
        return options

    def _sample_likelihood(self, state: torch.Tensor) -> torch.Tensor:
        result = self.model.sample_likelihood(state)
        return torch.as_tensor(result, dtype=torch.double).flatten()


__all__ = ["StochasticMapFilter"]
