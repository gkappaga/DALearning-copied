#!/usr/bin/env python3
"""
plot_training_curves.py

Loads one or more training record files (.pt) and produces combined:
  1) Training loss vs. epoch (all runs on one plot)
  2) Test RMSE vs. epoch (all runs on one plot)

Legend labels drop the leading timestamp from the folder name,
and the legend is placed outside to the right.
Output filenames are timestamped to avoid overwriting.

Usage:
  python plot_training_curves.py \
      --input save/.../2025-05-25_19-07lorenz96_1.0_10_60_8192_nl2_EnST_joint_LearnK/training_records.pt \
      --input save/.../another_run/training_records.pt \
      --output_dir figures
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

def extract_label(path_str):
    """
    Given the full path to training_records.pt, take its parent folder name
    and strip off any leading YYYY-MM-DD_HH-MM timestamp, returning a concise label.
    """
    folder = Path(path_str).parent.name
    # Remove leading timestamp like '2025-05-25_19-07'
    return re.sub(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}', '', folder)

def make_unique_name(base: str) -> str:
    """Append current timestamp to base name and add .png extension."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{ts}.png"

def load_records(paths):
    """
    Load each .pt file (onto CPU), auto-detect 'train_loss' and 'test_rmse'
    or fall back to the first two sequence-like entries.
    Returns dict: { label: { 'train_loss': [...], 'test_rmse': [...] } }
    """
    data = {}
    for p in paths:
        recs = torch.load(p, map_location="cpu")
        # find keys whose values are list/tuple/tensor-like (but not strings)
        seq_keys = [
            k for k, v in recs.items()
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes))
        ]
        if "train_loss" in recs and "test_rrmse" in recs:
            train_loss = recs["train_loss"]
            test_rmse  = recs["test_rrmse"]
        elif len(seq_keys) >= 2:
            train_loss = recs[seq_keys[0]]
            test_rmse  = recs[seq_keys[1]]
            print(f"Auto-detected keys for {p}: train_loss←'{seq_keys[0]}', test_rrmse←'{seq_keys[1]}'")
        else:
            raise KeyError(
                f"Could not find train_loss/test_rmse in {p}; available keys: {list(recs.keys())}"
            )
        label = extract_label(p)
        entry = {
            "train_loss": train_loss,
            "test_rrmse": test_rmse
        }

        # Add mean-penalty & cov-penalty if present
        if "train_mean_pen" in recs:
            entry["train_mean_pen"] = recs["train_mean_pen"]
        if "train_cov_pen" in recs:
            entry["train_cov_pen"] = recs["train_cov_pen"]

        data[label] = entry
    return data

def plot_and_save(all_data, series_key, ylabel, title, out_path):
    """
    Generic plotting routine:
      - all_data: dict of { label: { 'train_loss': [...], 'test_rrmse': [...] } }
      - series_key: either 'train_loss' or 'test_rrmse'
      - ylabel/title: axis label and title
      - out_path: Path where to save
    """
    plt.figure(figsize=(12, 6))
    for label, rec in all_data.items():
        y = rec[series_key]
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        plt.plot(
            y,
            label=label,
            linewidth=2
        )
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    parser = argparse.ArgumentParser("Plot combined training curves")
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="One or more paths to training_records.pt files"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=".",
        help="Directory to save the combined plots"
    )
    parser.add_argument(
        "--mc_penalty",
        default = False,
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    all_data = load_records(args.input)

    # Generate timestamped filenames
    loss_file = out_dir / make_unique_name("training_loss_all")
    rmse_file = out_dir / make_unique_name("test_rrmse_all")
    print(all_data.items())

    # Plot training loss
    plot_and_save(
        all_data,
        series_key="train_loss",
        ylabel="Training Loss",
        title="Training Loss vs. Epoch (All Runs)",
        out_path=loss_file
    )

    # Plot test RMSE
    plot_and_save(
        all_data,
        series_key="test_rrmse",
        ylabel="Test RRMSE",
        title="Test RRMSE vs. Epoch (All Runs)",
        out_path=rmse_file
    )
    if args.mc_penalty:
        #plot mean penalty
        plot_and_save(all_data,
                    series_key="train_mean_pen",
                    ylabel="Mean Penalty",
                    title="Mean Penalty vs. Epoch (All Runs)",
                    out_path=out_dir / make_unique_name("mean_penalty_all"))
        #plot covariance penalty
        plot_and_save(all_data,
                    series_key="train_cov_pen",
                    ylabel="Covariance Penalty",
                    title="Covariance Penalty vs. Epoch (All Runs)",
                    out_path=out_dir / make_unique_name("cov_penalty_all"))

if __name__ == "__main__":
    main()
