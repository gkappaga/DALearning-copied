from __future__ import annotations

from typing import Callable

import torch


def rk4(step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state: torch.Tensor,
        time: torch.Tensor,
        dt: float) -> torch.Tensor:
    """
    Fourth-order Runge-Kutta integrator matching the MATLAB ``rk4`` helper.

    Parameters
    ----------
    step_fn:
        Callable ``f(x, t, dt)`` returning time derivative at ``(x, t)``.
    state:
        Current state vector (``(..., d)`` tensor).
    time:
        Current simulation time (scalar tensor).
    dt:
        Time step.
    """
    k1 = step_fn(time, state)
    k2 = step_fn(time + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = step_fn(time + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = step_fn(time + dt, state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


__all__ = ["rk4"]
