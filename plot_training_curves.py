#!/usr/bin/env python3
"""
plot_training_curves.py

Loads one or more training record files (.pt) and produces combined:
  1) Training loss vs. epoch (all runs on one plot)
  2) Test RRMSE vs. epoch (all runs on one plot)

If --mc_penalty is set and the records contain
train_mean_pen, train_cov_pen, train_orig_pen, it will also plot:
  3) Mean penalty vs. epoch
  4) Covariance penalty vs. epoch
  5) Original penalty vs. epoch
  6) A combined plot of all four series on one figure.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re
from collections.abc import Sequence
import numpy as np

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item() if x.dim()==0 else x.detach().cpu().numpy()
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return np.array([ to_numpy(el) for el in x ])
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def extract_label(path_str):
    folder = Path(path_str).parent.name
    return re.sub(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}', '', folder)

def make_unique_name(base: str) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{ts}.png"

def load_records(paths):
    data = {}
    for p in paths:
        raw = torch.load(p, map_location="cpu")
        recs = {}
        for k, v in raw.items():
            recs[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v

        seq_keys = [
            k for k,v in recs.items()
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes))
        ]

        # primary series
        if "train_loss" not in recs or "test_rrmse" not in recs:
            raise KeyError(f"Need train_loss & test_rrmse in {p}; got {list(recs.keys())}")
        entry = {
            "train_loss": recs["train_loss"],
            "test_rrmse": recs["test_rrmse"]
        }

        # optional penalties
        for pen in ("train_mean_pen", "train_cov_pen", "train_orig_pen"):
            if pen in recs:
                entry[pen] = recs[pen]

        data[ extract_label(p) ] = entry
    return data

def plot_and_save(all_data, series_key, ylabel, title, out_path):
    plt.figure(figsize=(10,5))
    for label, rec in all_data.items():
        y = to_numpy(rec[series_key])
        plt.plot(y, label=label, linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02,1), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

def plot_multiple(all_data, series_keys, ylabel, title, out_path):
    plt.figure(figsize=(10,5))
    for label, rec in all_data.items():
        for key in series_keys:
            if key in rec:
                y = to_numpy(rec[key])
                plt.plot(y, label=f"{label}:{key}", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02,1), frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plot to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i","--input", nargs="+", required=True,
        help="Paths to training_records.pt"
    )
    parser.add_argument(
        "-o","--output_dir", default=".",
        help="Where to save the plots"
    )
    parser.add_argument(
        "--mc_penalty",
        help="Also look for and plot mean/cov/orig penalties"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_records(args.input)

    # 1) training loss
    plot_and_save(
        all_data,
        series_key="train_loss",
        ylabel="Training Loss",
        title="Training Loss vs. Epoch",
        out_path=out_dir / make_unique_name("training_loss_all")
    )

    # 2) test RRMSE
    plot_and_save(
        all_data,
        series_key="test_rrmse",
        ylabel="Test RRMSE",
        title="Test RRMSE vs. Epoch",
        out_path=out_dir / make_unique_name("test_rrmse_all")
    )

    if args.mc_penalty:
        # individual penalty plots
        for pen_key, pen_name in [
            ("train_mean_pen", "Mean Penalty"),
            ("train_cov_pen", "Covariance Penalty"),
            ("train_orig_pen", "Original Penalty"),
        ]:
            # only if present
            if any(pen_key in rec for rec in all_data.values()):
                plot_and_save(
                    all_data,
                    series_key=pen_key,
                    ylabel=pen_name,
                    title=f"{pen_name} vs. Epoch",
                    out_path=out_dir / make_unique_name(pen_key)
                )

        # combined all four on one plot
        combined_keys = ["train_loss", "train_orig_pen", "train_mean_pen", "train_cov_pen"]
        plot_multiple(
            all_data,
            series_keys=combined_keys,
            ylabel="Value",
            title="Loss & Penalties vs. Epoch",
            out_path=out_dir / make_unique_name("loss_and_penalties_all")
        )

if __name__=="__main__":
    main()
