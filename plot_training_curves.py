#!/usr/bin/env python3
"""
plot_training_curves.py

Loads one or more training record files (.pt) and produces combined:
  1) Training loss vs. epoch (all runs on one plot)
  2) Test RMSE vs. epoch (all runs on one plot)

Legend is placed outside to the right.  
Output files are timestamped to avoid overwriting.

Usage:
  python plot_training_curves.py \
      --input save/.../run1/training_records.pt save/.../run2/training_records.pt \
      --output_dir figures
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def extract_label(path_str):
    # Use the immediate parent directory name as the legend label
    return Path(path_str).parent.name

def make_unique_name(base):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{ts}.png"

def load_records(paths):
    data = {}
    for p in paths:
        records = torch.load(p, map_location='cpu')
        # Collect all sequence-like keys
        seq_keys = [k for k, v in records.items()
                    if hasattr(v, '__len__') and not isinstance(v, (str, bytes))]
        # Pick the right fields
        if 'train_loss' in records and 'test_rmse' in records:
            train_loss = records['train_loss']
            test_rmse  = records['test_rmse']
        elif len(seq_keys) >= 2:
            train_loss = records[seq_keys[0]]
            test_rmse  = records[seq_keys[1]]
            print(f"Auto-detected {p}: train_loss←'{seq_keys[0]}', test_rmse←'{seq_keys[1]}'")
        else:
            raise KeyError(
                f"Cannot find train_loss/test_rmse in {p}. Keys: {list(records.keys())}"
            )
        label = extract_label(p)
        data[label] = {'train_loss': train_loss, 'test_rmse': test_rmse}
    return data

def main():
    parser = argparse.ArgumentParser(description="Plot combined training curves")
    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        required=True,
        help='One or more paths to training_records.pt'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='.',
        help='Directory to save the plots'
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_records(args.input)

    # Prepare unique filenames
    loss_fname = make_unique_name('training_loss_all')
    rmse_fname = make_unique_name('test_rmse_all')

    # --- Plot Training Loss ---
    plt.figure(figsize=(12, 6))                           # wider figure
    for label, rec in all_data.items():
        plt.plot(
            range(1, len(rec['train_loss']) + 1),
            rec['train_loss'],
            label=label,
            linewidth=2                                  # a bit thicker lines
        )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.title('Training Loss vs. Epoch (All Runs)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)             # dashed, lighter grid
    # slimmer legend outside
    leg = plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # --- Plot Test RMSE ---
    plt.figure(figsize=(12, 6))
    for label, rec in all_data.items():
        plt.plot(
            range(1, len(rec['test_rmse']) + 1),
            rec['test_rmse'],
            label=label,
            linewidth=2
        )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test RMSE', fontsize=14)
    plt.title('Test RMSE vs. Epoch (All Runs)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    leg = plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(rmse_path, dpi=300)
    plt.close()

    print(f"Saved combined test RMSE plot to {rmse_path}")

if __name__ == '__main__':
    main()
