#!/usr/bin/env python3
"""
plot_training_curves.py

Loads one or more training record files (.pt) and produces combined:
  1) Training loss vs. epoch (all runs on one plot)
  2) Test RMSE vs. epoch (all runs on one plot)

Legend is placed outside to the right, lines are thicker, grid is dashed,
and output filenames are timestamped to avoid overwriting.

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
    return Path(path_str).parent.name

def make_unique_name(base):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{ts}.png"

def load_records(paths):
    data = {}
    for p in paths:
        rec = torch.load(p, map_location='cpu')
        # auto-detect keys if needed
        keys = [k for k,v in rec.items() if hasattr(v, '__len__') and not isinstance(v, (str,bytes))]
        if 'train_loss' in rec and 'test_rmse' in rec:
            train_loss = rec['train_loss']
            test_rmse  = rec['test_rmse']
        elif len(keys) >= 2:
            train_loss = rec[keys[0]]
            test_rmse  = rec[keys[1]]
            print(f"Auto-detected in {p}: train_loss←'{keys[0]}', test_rmse←'{keys[1]}'")
        else:
            raise KeyError(f"Missing train_loss/test_rmse in {p}, got {list(rec.keys())}")
        label = extract_label(p)
        data[label] = {'train_loss': train_loss, 'test_rmse': test_rmse}
    return data

def main():
    p = argparse.ArgumentParser("Plot combined training curves")
    p.add_argument('-i','--input', nargs='+', required=True,
                   help='Paths to training_records.pt files')
    p.add_argument('-o','--output_dir', default='.', help='Where to save PNGs')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_records(args.input)

    # prepare unique filenames
    loss_path = out_dir / make_unique_name('training_loss_all')
    rmse_path = out_dir / make_unique_name('test_rmse_all')

    # --- Plot Training Loss ---
    plt.figure(figsize=(12,6))
    for label, rec in all_data.items():
        plt.plot(
            range(1, len(rec['train_loss'])+1),
            rec['train_loss'],
            label=label,
            linewidth=2
        )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.title('Training Loss vs. Epoch (All Runs)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01,1),
               frameon=False, fontsize=12)
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved combined training loss plot to {loss_path}")

    # --- Plot Test RMSE ---
    plt.figure(figsize=(12,6))
    for label, rec in all_data.items():
        plt.plot(
            range(1, len(rec['test_rmse'])+1),
            rec['test_rmse'],
            label=label,
            linewidth=2
        )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test RMSE', fontsize=14)
    plt.title('Test RMSE vs. Epoch (All Runs)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01,1),
               frameon=False, fontsize=12)
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(rmse_path, dpi=300)
    plt.close()
    print(f"Saved combined test RMSE plot to {rmse_path}")

if __name__ == '__main__':
    main()
