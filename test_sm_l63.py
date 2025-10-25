from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from stochastic_maps_py.data_assimilation import post_process, seq_assimilation
from stochastic_maps_py.examples import build_lorenz63_problem
from stochastic_maps_py.methods import StochasticMapFilter


def main() -> None:
    # keep the run reasonably short so it finishes quickly
    model, options = build_lorenz63_problem(
        T_spin_up=200,
        T_steps=400,
        ensemble_size=128,
        seed=42,
    )

    sm_filter = StochasticMapFilter(model, options)
    assimilation = seq_assimilation(model, sm_filter.sample_posterior)

    stats = post_process(model, assimilation, options)
    print("RMSE mean:", float(stats["RMSE_mean"]))
    print("RMSE std:", float(stats["RMSE_std"]))
    print("Spread mean:", float(stats["spread_mean"]))
    print("Coverage probability:", float(stats["covprob_mean"]))

    start_idx = int(round(model.t0 / (model.dt * model.dt_iter)))
    steps = model.J
    truth = model.xt[start_idx : start_idx + steps].detach().cpu().numpy()
    ensemble_mean = assimilation.mean.detach().cpu().numpy()
    times = assimilation.time.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    state_labels = ["x", "y", "z"]
    for idx, ax in enumerate(axes):
        ax.plot(times, truth[:, idx], label="Truth", linewidth=1.2)
        ax.plot(times, ensemble_mean[:, idx], label="Ensemble mean", linewidth=1.2)
        ax.set_ylabel(state_labels[idx])
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time")
    fig.suptitle("Lorenz-63: Truth vs. Stochastic Map Ensemble Mean")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / "lorenz63_truth_vs_mean.png"
    fig.savefig(fig_path, dpi=150)
    print("Saved figure to", fig_path)


if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    main()
