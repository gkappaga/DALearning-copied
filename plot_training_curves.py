#!/usr/bin/env python3
"""
plot_training_curves.py

Loads one or more training record files (.pt) and produces combined:
  1) Training loss vs. epoch
  2) Test RMSE vs. epoch

Files are saved with a timestamp suffix to avoid overwriting.

Usage:
  python plot_training_curves.py --input save/.../run1/training_records.pt save/.../run2/training_records.pt \
      --output_dir figures
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def extract_label(path_str):
    return Path(path_str).parent.name

def load_records(paths):
    data = {}
    for p in paths:
        records = torch.load(p, map_location='cpu')
        train_loss = records.get('train_loss')
        test_rmse = records.get('test_rmse')
        if train_loss is None or test_rmse is None:
            raise KeyError(f"Expected keys 'train_loss' and 'test_rmse' in {p}")
        label = extract_label(p)
        data[label] = {
            'train_loss': train_loss,
            'test_rmse': test_rmse
        }
    return data

def make_unique_name(base):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{ts}.png"

def main():
    parser = argparse.ArgumentParser(description="Plot combined training curves")
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='One or more paths to training_records.pt files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the combined plots'
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_records(args.input)

    # Prepare filenames
    loss_fname = make_unique_name('training_loss_all')
    rmse_fname = make_unique_name('test_rmse_all')

    # Plot training loss
    plt.figure(figsize=(8,6))
    for label, rec in all_data.items():
        plt.plot(
            range(1, len(rec['train_loss']) + 1),
            rec['train_loss'],
            label=label
        )
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epoch (All Runs)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    loss_path = out_dir / loss_fname
    plt.savefig(loss_path, dpi=300)
    print(f"Saved combined training loss plot to {loss_path}")

    # Plot test RMSE
    plt.figure(figsize=(8,6))
    for label, rec in all_data.items():
        plt.plot(
            range(1, len(rec['test_rmse']) + 1),
            rec['test_rmse'],
            label=label
        )
    plt.xlabel('Epoch')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE vs. Epoch (All Runs)')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    rmse_path = out_dir / rmse_fname
    plt.savefig(rmse_path, dpi=300)
    print(f"Saved combined test RMSE plot to {rmse_path}")

if __name__ == '__main__':
    main()
