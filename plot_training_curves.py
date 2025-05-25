#!/usr/bin/env python3
"""
plot_training_curves.py

Loads training records (saved as a .pt file) and produces:
  1) Training loss vs. epoch
  2) Test RMSE vs. epoch

Usage:
  python plot_training_curves.py --input training_records.pt
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "--input",
        type=str,
        default="training_records.pt",
        help="Path to the .pt file containing training records"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the plots"
    )
    args = parser.parse_args()

    # Load the records onto CPU (no CUDA needed)
    records = torch.load(args.input, map_location="cpu")

    # Expect records to be a dict with keys 'train_loss' and 'test_rmse'
    train_loss = records.get("train_loss")
    test_rmse = records.get("test_rrmse")

    if train_loss is None or test_rmse is None:
        raise KeyError("Expected keys 'train_loss' and 'test_rmse' in the loaded records")

    epochs = range(1, len(train_loss) + 1)

    # Plot training loss
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epoch")
    plt.grid(True)
    plt.tight_layout()
    loss_path = Path(args.output_dir) / "training_loss_curve.png"
    plt.savefig(loss_path)
    print(f"Saved training loss curve to {loss_path}")

    # Plot test RMSE
    plt.figure()
    plt.plot(epochs, test_rmse)
    plt.xlabel("Epoch")
    plt.ylabel("Test RMSE")
    plt.title("Test RMSE vs. Epoch")
    plt.grid(True)
    plt.tight_layout()
    rmse_path = Path(args.output_dir) / "test_rmse_curve.png"
    plt.savefig(rmse_path)
    print(f"Saved test RMSE curve to {rmse_path}")

if __name__ == "__main__":
    main()
