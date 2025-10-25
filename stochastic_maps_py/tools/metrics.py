from __future__ import annotations

import torch


def lorenz_distance_matrix(dim_state: int) -> torch.Tensor:
    """
    Compute the cyclic pairwise distance matrix used in the MATLAB implementation.

    Parameters
    ----------
    dim_state:
        Dimension of the Lorenz state vector.

    Returns
    -------
    torch.Tensor
        ``(dim_state, dim_state)`` integer tensor with pairwise distances on a ring.
    """
    if dim_state <= 0:
        raise ValueError("`dim_state` must be positive.")

    dist = torch.zeros((dim_state, dim_state), dtype=torch.int64)
    indices = torch.arange(dim_state)
    for i in range(dim_state):
        delta = torch.abs(indices - i)
        wrap = torch.where(indices < i, indices + dim_state - i, i + dim_state - indices)
        dist[i] = torch.minimum(delta, wrap)
    return dist


__all__ = ["lorenz_distance_matrix"]
