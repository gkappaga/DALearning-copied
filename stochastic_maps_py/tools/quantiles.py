from __future__ import annotations

from typing import Iterable, Sequence, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float], Iterable[float]]


def _to_tensor(vector: TensorLike, *, dtype: torch.dtype = torch.double) -> torch.Tensor:
    """Convert input to a 1-D tensor of the desired dtype."""
    tensor = torch.as_tensor(vector, dtype=dtype)
    if tensor.ndim != 1:
        raise ValueError("`vector` must be one-dimensional.")
    return tensor


def quantiles_sorted_vector(vector: TensorLike, probs: Union[int, TensorLike]) -> torch.Tensor:
    """
    Replicates the behaviour of the MATLAB ``quantiles_sorted_vector`` helper.

    Parameters
    ----------
    vector:
        Sorted one-dimensional samples. The function does *not* sort the input.
    probs:
        Either:
            * an integer ``m > 1`` requesting ``m`` equally spaced quantiles, or
            * an iterable of probabilities in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        1-D tensor containing the requested quantiles (double precision).
    """

    samples = _to_tensor(vector)
    n = samples.numel()
    if n == 0:
        raise ValueError("`vector` must contain at least one value.")

    if isinstance(probs, int):
        if probs <= 1:
            raise ValueError("Integer `probs` must be greater than one.")
        p = torch.linspace(1, probs, probs, dtype=torch.double) / (probs + 1)
    else:
        p = _to_tensor(probs)
        if torch.any((p < 0) | (p > 1)):
            raise ValueError("Quantile probabilities must lie in [0, 1].")

    r = p * n
    k = torch.floor(r + 0.5).to(torch.long)
    kp1 = k + 1
    r = r - k

    k = torch.clamp(k, min=1)
    kp1 = torch.clamp(kp1, max=n)

    # Convert to zero-based indexing.
    k0 = k - 1
    kp10 = kp1 - 1

    values = (0.5 + r) * samples[kp10] + (0.5 - r) * samples[k0]
    return values.reshape(1, -1).squeeze(0)


__all__ = ["quantiles_sorted_vector"]
