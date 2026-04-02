import re
import os
import os, torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # may throw if an op has no det. path

# Turn off TF32 to avoid GEMM path changes
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# CUBLAS determinism for matmuls (set before import torch if possible)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"

import math
import sys
import copy
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from contextlib import contextmanager

from train_test_utils_v2 import test_ClassicFilter_v2
from utils import partial_obs_operator, get_dataloader, redirect_output
from config.cli import get_parameters
from config.dataset_info import DATASET_INFO

# dapper
import dapper as dpr 
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.tools.localization import nd_Id_localization
from dapper.mods.Lorenz96 import LPs

# customized
from grid_search_config import GRID_SEARCH_INFO

@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull  # Redirect stdout to null
            sys.stderr = devnull  # Redirect stderr to null
            yield
        finally:
            sys.stdout = old_stdout  # Restore stdout
            sys.stderr = old_stderr  # Restore stderr

def print_run_signature(tag, args):
    print(f"\n[{tag}] RUN SIGNATURE")
    keys = [
        "dataset","v","N","seed",
        "random_h","random_noise","access_to_H","access_to_noise",
        "ori_dim","obs_dim","sigma_y",
        "test_traj_num","test_batch_size","test_steps",
        "cp_load_path",
    ]
    for k in keys:
        print(f"  {k}: {getattr(args, k, None)}")
    oi = getattr(args, "obs_inds", None)
    if oi is not None:
        oi_cpu = oi.detach().cpu() if hasattr(oi, "detach") else oi
        oi_list = list(oi_cpu[:10]) if len(oi_cpu) >= 10 else list(oi_cpu)
        print(f"  obs_inds[:10] (len={len(oi_cpu)}): {oi_list}")

def plot_heatmap_with_nan(data, x_list, y_list, save_path=None, img_title="Grid Search"):
    """
    Plots a heatmap with NaN values handled and saves it to a specified path.

    Parameters:
        data (numpy.ndarray): 2D array of data values (NaN values are allowed).
        x_list (list or numpy.ndarray): List of values for the x-axis (corresponding to rows of data).
        y_list (list or numpy.ndarray): List of values for the y-axis (corresponding to columns of data).
        save_path (str): File path to save the plot.

    Returns:
        None
    """
    # Validate dimensions
    if data.shape[0] != len(x_list):
        raise ValueError("The length of x_list must match the number of rows in data.")
    if data.shape[1] != len(y_list):
        raise ValueError("The length of y_list must match the number of columns in data.")

    # Replace NaN with a value greater than the maximum
    max_value = np.nanmax(data)
    nan_value = max_value * 1.1  # Set NaN to 1.1 times the max value
    data_with_nan_replaced = np.where(np.isnan(data), nan_value, data)

    # Create grid edges (for pcolormesh)
    x_edges = np.linspace(x_list[0] - (x_list[1] - x_list[0]) / 2, 
                          x_list[-1] + (x_list[1] - x_list[0]) / 2, len(x_list) + 1)
    y_edges = np.linspace(y_list[0] - (y_list[1] - y_list[0]) / 2, 
                          y_list[-1] + (y_list[1] - y_list[0]) / 2, len(y_list) + 1)

    # Use the 'jet' colormap
    cmap = plt.cm.jet

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(y_edges, x_edges, data_with_nan_replaced, cmap=cmap, shading='auto')

    # Add a colorbar
    cbar = plt.colorbar(mesh)
    cbar.set_label("Values", fontsize=15)

    # Adjust colorbar ticks to include NaN
    cbar_ticks = np.linspace(np.nanmin(data), max_value, num=6)  # Generate ticks for original data range
    cbar_ticks = np.append(cbar_ticks, nan_value)  # Add NaN as the last tick
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar_ticks[:-1]] + ["NaN"])  # Add "NaN" label
    cbar.ax.tick_params(labelsize=15)

    # Explicitly set ticks and labels
    plt.xticks(ticks=y_list, 
               labels=[f"{y_list[0]:.0e}"] + [f"{val:.0f}" for val in y_list[1:]], 
               fontsize=15,
               rotation=45)
    plt.yticks(ticks=x_list, 
               labels=[f"{val:.2f}" for val in x_list], 
               fontsize=15)

    # Label axes
    plt.xlabel("Localization Radius", fontsize=15)
    plt.ylabel("Inflation Factors", fontsize=15)

    # Set title
    plt.title(img_title, fontsize=18)

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Save plot to file
    plt.close()  # Close the plot to release memory

def plot_simple(x, y, save_path=None, img_title="Infl Search"):
    """
    Plots a simple 2D line plot for two arrays, handling NaN values.

    Parameters:
        x (list or numpy.ndarray): Array of x-axis values.
        y (list or numpy.ndarray): Array of y-axis values.
        save_path (str): File path to save the plot. If None, plot is shown instead.
        img_title (str): Title of the plot. Default is "Infl Search".

    Returns:
        None
    """
    # Convert inputs to numpy arrays if they aren't already
    x = np.array(x)
    y = np.array(y)

    # Check if y contains NaN values
    if np.all(np.isnan(y)):  # All values are NaN
        y = np.zeros_like(y)  # Replace all NaN with 0
        special_ticks = {0: "NaN"}  # Add 0 as a tick labeled "NaN"
        max_value = 0  # No valid max value
    elif np.any(np.isnan(y)):  # Some values are NaN
        max_non_nan = np.nanmax(y)  # Get maximum non-NaN value
        special_value = max_non_nan * 1.3
        nan_indices = np.where(np.isnan(y))  # Indices of NaN values
        y[nan_indices] = special_value  # Replace NaN with 1.3 * max
        special_ticks = {special_value: "NaN"}  # Add special value to ticks
        max_value = max_non_nan
    else:
        special_ticks = {}  # No NaN, no special ticks
        max_value = np.max(y)

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')  # Line with markers

    # Add labels and title
    plt.title(img_title, fontsize=18)
    plt.xlabel("Inflation Factors", fontsize=15)
    plt.ylabel("RMSE", fontsize=15)

    # Customize y-ticks to only show up to slightly beyond max_value
    if special_ticks:
        max_tick_value = max_value * 1.05  # Allow a bit of padding beyond max value
        current_ticks = plt.yticks()[0]  # Get current y-ticks
        filtered_ticks = [tick for tick in current_ticks if tick <= max_tick_value]  # Filter ticks
        filtered_ticks = np.append(filtered_ticks, list(special_ticks.keys()))  # Add special ticks
        plt.yticks(
            filtered_ticks,
            labels=[
                f"{tick:.2f}" if tick not in special_ticks else special_ticks[tick]
                for tick in filtered_ticks
            ],
            fontsize=15
        )
    else:
        plt.yticks(fontsize=15)

    # Customize x-ticks
    plt.xticks(fontsize=15, rotation=45, ha='right')

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Save plot to file
    else:
        plt.show()  # Show plot interactively
    plt.close()  # Close the plot to release memory


def main(grid_search_info):
    base_args = get_parameters()          # defaults only
    base_args.test_only = True
    base_args.seed = 42
    base_args.random_noise = True
    base_args.access_to_noise = True
    base_args.access_to_H = True
    base_args.test_batch_size = 64
    base_args.test_traj_num = 64
    base_args.random_h = True
    base_args.cp_load_path = 'no'

    def make_args(**overrides):
        a = copy.deepcopy(base_args)
        for k, v in overrides.items():
            setattr(a, k, v)
        return a
    
    # def set_all_seeds(seed: int):
    #     # random.seed(seed)
    #     # np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)

    def run_one_trial(base_args, cfg, trial_ind):
        args = copy.deepcopy(base_args)

        # trial seed (applies to dataloader shuffling + model randomness)
        # seed = int(base_args.seed)
        # args.seed = seed
        # set_all_seeds(seed)

        # apply sweep params
        args.N = cfg["N"]
        args.v = cfg["method"]          # if your code uses args.v to choose filter
        # args.dataset and args.sigma_y should already be set on base_args for this loop
        infl = cfg["infl"]
        # infl = 1.1
        loc_radius = cfg["loc_rad"]
        args.dataset = cfg["dataset"]
        args.ori_dim = cfg["ori_dim"]
        args.obs_dim = cfg["obs_dim"]
        args.random_h = True
        args.random_noise = True
        args.cp_load_path = 'no'  # No checkpoint loading for this test
        args.seed = 42
        args.test_only = True
        args.test_traj_num = 64
        args.test_batch_size = 64
        # args.N = 20
        # args.v = 'ESRF'
        args.obs_inds = torch.arange(0, args.ori_dim, args.ori_dim // args.obs_dim)
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)
        print_run_signature('grid', args)
        # Call your code.
        # loader param is unused in your snippet since the function makes its own test_loader.
        test_loader = get_dataloader(args, test_only = True)
        print(f"Running trial {trial_ind} with N={args.N}, infl={infl}, loc_radius={loc_radius}, sigma_y={args.sigma_y}, dataset = {args.dataset}, v= {args.v}")
        out = test_ClassicFilter_v2(
            loader=test_loader,
            args=args,
            plot=False,
            H_info=H_info,
            infl=infl,
            loc_radius=loc_radius,
            plot_figures=False,
            save_pdf=True,
        )
        # test_ClassicFilter_v2(test_loader, args, plot=False, H_info=H_info, plot_figures=False, 
        # fig_name=f'{folder_name}/test_{args.N}', save_pdf=True, infl=1.1, loc_radius=None)
        # Normalize return shape to dict
        rmse  = out.get("mean_rmse", float("nan"))
        rmse_std = out.get("std_rmse", float("nan"))
        rmv   = out.get("mean_rmv", float("nan"))
        rmv_std  = out.get("std_rmv", float("nan"))
        rrmse = out.get("mean_rrmse", float("nan"))
        rrmse_std = out.get("std_rrmse", float("nan"))
        crps = out.get("mean_crps", float("nan"))
        crps_std = out.get("std_crps", float("nan"))
        rcrps = out.get("mean_rcrps", float("nan"))
        rcrps_std = out.get("std_rcrps", float("nan"))
        print(rrmse)
        return rmse, rmse_std, rmv, rmv_std, rrmse, rrmse_std, crps, crps_std, rcrps, rcrps_std

        # If it returns a tuple like (mean_rrmse, mean_rmse, ...)
        # adjust this to match your actual return order:
        mean_rrmse, mean_rmse = out[0], out[1]
        return mean_rmse, np.nan, mean_rrmse
    def _process_trial(trial_ind):
        rmse_mean  = np.full(K, np.nan)
        rmse_std   = np.full(K, np.nan)
        rmv_mean   = np.full(K, np.nan)
        rmv_std    = np.full(K, np.nan)
        rrmse_mean = np.full(K, np.nan)
        rrmse_std  = np.full(K, np.nan)
        crps_mean = np.full(K, np.nan)
        crps_std = np.full(K, np.nan)
        rcrps_mean = np.full(K, np.nan)
        rcrps_std = np.full(K, np.nan)

        for k, cfg in enumerate(experiments):
            # with suppress_output():
            rmse, rmse_s, rmv, rmv_s, rrmse, rrmse_s, crps, crps_s, rcrps, rcrps_s = run_one_trial(base_args, cfg, trial_ind)

            rmse_mean[k]  = rmse
            rmse_std[k]   = rmse_s
            rmv_mean[k]   = rmv
            rmv_std[k]    = rmv_s
            rrmse_mean[k] = rrmse
            rrmse_std[k]  = rrmse_s
            crps_mean[k] = crps
            crps_std[k] = crps_s
            rcrps_mean[k] = rcrps
            rcrps_std[k] = rcrps_s

        return rmse_mean, rmse_std, rmv_mean, rmv_std, rrmse_mean, rrmse_std, crps_mean, crps_std, rcrps_mean, rcrps_std

        
        
        # try:
        #     # Find the first key that contains 'rmsea'
        #     rmse_a_key = [key for key in averages.keys() if 'rmse.a' in key][0]
        # except Exception as e:
        #     # If not found, print available keys
        #     print("Exception message:", e)
        #     print("Available keys:", averages.keys())

        # try:
        #     # Find the first key that contains 'rmva'
        #     rmv_a_key = [key for key in averages.keys() if 'rmv.a' in key][0]
        # except Exception as e:
        #     # If not found, log error details and set rmv_a_key to rmse_a_key
        #     print("Exception message:", e)
        #     print(f"Error trial: {method_name} on {dataset} with sigma_y = {sigma_y}, ensemble size = {N}")
        #     print("Available keys:", averages.keys())
        #     rmv_a_key = rmse_a_key



        # rmse_mean, rmse_std, rmv_mean, rmv_std = [], [], [], []

        # for entry in averages[rmse_a_key]:
        #     clean_entry = entry.replace('␣', '').strip()
        #     if 'nan' in clean_entry.lower():
        #         rmse_mean.append(float('nan'))
        #         rmse_std.append(float('nan'))                                  
        #     else:
        #         match = re.match(r'([0-9.]+)\s*±\s*([0-9.]+)', clean_entry)
        #         if match:
        #             rmse_mean.append(float(match.group(1)))
        #             rmse_std.append(float(match.group(2)))
        
        # # Convert lists to numpy arrays for efficient computation
        # rmse_mean = np.array(rmse_mean)
        # rmse_std = np.array(rmse_std)
                    
        # for entry in averages[rmv_a_key]:
        #     clean_entry = entry.replace('␣', '').strip()
        #     if 'nan' in clean_entry.lower():
        #         rmv_mean.append(float('nan'))
        #         rmv_std.append(float('nan'))
        #     else:
        #         match = re.match(r'([0-9.]+)\s*±\s*([0-9.]+)', clean_entry)
        #         if match:
        #             rmv_mean.append(float(match.group(1)))
        #             rmv_std.append(float(match.group(2)))
        
        # # Convert rmv_mean and rmv_std to numpy arrays
        # rmv_mean = np.array(rmv_mean)
        # rmv_std = np.array(rmv_std)
        
        # rrmse_mean = rmse_mean / obs_states_rms
        # rrmse_std = rmse_std / obs_states_rms
        
        # return rmse_mean, rmse_std, rmv_mean, rmv_std, rrmse_mean, rrmse_std


    def _combine_results(results):
        len_data = len(results[0])
        
        result_record = np.zeros((GRID_SEARCH_INFO[dataset]["num_trials"], len_data, K))

        for trial_ind, data in enumerate(results):
            for i in range(len_data):
                result_record[trial_ind, i, :] = data[i]
        
        valid_val_count = np.sum(~np.isnan(result_record), axis=0)
        safe_valid_val_count = np.where(valid_val_count == 0, 1, valid_val_count)
        
        # Store original result_record with NaNs
        result_record_original = result_record.copy()
        
        # Replace NaNs with zeros for mean calculation
        result_record[np.isnan(result_record)] = 0
        result_avg = np.sum(result_record, axis=0) / safe_valid_val_count
        result_avg[valid_val_count == 0] = np.nan
        
        # Calculate standard deviation using numpy's nanstd function
        result_std = np.nanstd(result_record_original, axis=0, ddof=1)
        
        return result_avg, result_std, valid_val_count[0, :]

    def _check_trial_processed(dataset, method_name, N):
        file_path = os.path.join("save", f"benchmarks_random_h_noise_{dataset}.csv")

        if not os.path.exists(file_path):
            return False
    
        df = pd.read_csv(file_path)
        matching_rows = df[(df['method'] == method_name) & (df['N'] == N)]
        if not matching_rows.empty:
            return True
        else: 
            return False

    for dataset, info in grid_search_info.items():

        for method_name in info['methods']:

            N_list = info['N_list']
    
            for N in N_list:
                # Check the type of infl_list and loc_rad_list
                if method_name == "LETKF" and 'letkf_infl_list' in info:
                    # Use letkf_infl_list if available and method_name is "LETKF"
                    if isinstance(info['letkf_infl_list'], list):
                        infl_list = info['letkf_infl_list']
                    elif isinstance(info['letkf_infl_list'], dict):
                        infl_list = info['letkf_infl_list'].get(N, [])
                else:
                    # Fallback to infl_list
                    if isinstance(info['infl_list'], list):
                        infl_list = info['infl_list']
                    elif isinstance(info['infl_list'], dict):
                        infl_list = info['infl_list'].get(N, [])
                
                if method_name == 'LETKF' or method_name == 'EnKF':
                    if isinstance(info['loc_rad_list'], list):
                        loc_rad_list = info['loc_rad_list']
                    elif isinstance(info['loc_rad_list'], dict):
                        loc_rad_list = info['loc_rad_list'].get(N, [])
                else:
                    loc_rad_list = []
                
                trial_processed = _check_trial_processed(dataset, method_name, N)

                # xps = dpr.xpList()
                experiments = []
                for infl in infl_list:
                    if loc_rad_list:
                        for loc_rad in loc_rad_list:
                            experiments.append({
                                "method": method_name,
                                "N": N,
                                "infl": infl,
                                "loc_rad": loc_rad,
                                "dataset": dataset,
                                "ori_dim": DATASET_INFO[dataset]["dim"],
                                "obs_dim": DATASET_INFO[dataset]["obs_dim"]
                            })
                    else:
                        experiments.append({
                            "method": method_name,
                            "N": N,
                            "infl": infl,
                            "loc_rad": None,
                            "dataset": dataset,
                            "ori_dim": DATASET_INFO[dataset]["dim"],
                            "obs_dim": DATASET_INFO[dataset]["obs_dim"]
                        })

                if trial_processed:
                    print(f"Grid search results exist for {method_name} on {dataset} with random noise, ensemble size {N}.")
                    data_save_path = os.path.join('save', 'data', f'{dataset}_{method_name}_{N}_results.npz')
                    saved_data = np.load(data_save_path)

                    result_avg = saved_data['result_avg']
                    result_std = saved_data['result_std']
                    valid_val_count = saved_data['valid_val_count']
                    infl_list = saved_data['infl_list'].tolist()
                    loc_rad_list = saved_data['loc_rad_list'].tolist()
                else:
                    print(f"Start grid search for {method_name} on {dataset} with random noise, ensemble size {N}.")
                    
                    K = len(experiments)
                    
                    results = Parallel(n_jobs=1)(delayed(_process_trial)(trial_ind) for trial_ind in range(GRID_SEARCH_INFO[dataset]["num_trials"]))
                    result_avg, result_std, valid_val_count = _combine_results(results)

                    data_save_path = os.path.join('save', 'data', f'{dataset}_{method_name}_{N}_results.npz')
                    os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
                    np.savez(data_save_path, 
                                result_avg=result_avg, 
                                result_std=result_std,
                                valid_val_count=valid_val_count, 
                                infl_list=infl_list, 
                                loc_rad_list=loc_rad_list)

                if loc_rad_list:
                    rmse_avg_grid = result_avg[0, :].reshape(len(infl_list), len(loc_rad_list))
                    img_save_path = os.path.join('save', 'figures', f'{dataset}_{method_name}_{N}_gridsearch.png')
                    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                    img_title = f"{method_name} Grid Search with Ensemble Size {N}"
                    plot_heatmap_with_nan(rmse_avg_grid, infl_list, loc_rad_list, save_path=img_save_path, img_title=img_title)
                else:
                    rmse_avg_grid = result_avg[0, :]
                    img_save_path = os.path.join('save', 'figures', f'{dataset}_{method_name}_{N}_gridsearch.png')
                    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                    img_title = f"{method_name} Parameter Search with Ensemble Size {N}"
                    plot_simple(infl_list, rmse_avg_grid, save_path=img_save_path, img_title=img_title)
                    
                best_rmse_ind = np.nanargmin(result_avg[0, :])  # your rrmse_mean index
                best_cfg = experiments[best_rmse_ind]

                best_loc_rad = best_cfg["loc_rad"] if best_cfg["loc_rad"] is not None else ""
                best_infl = best_cfg["infl"]

                # Define lists of metrics and their types
                metric_names = ["mean_rmse", "std_rmse", "mean_rmv", "std_rmv", "mean_rrmse", "std_rrmse", "mean_crps", "std_crps", "mean_rcrps", "std_rcrps"]
                
                # Initialize an empty dictionary for DataFrame
                df_dict = {
                    "method": [method_name],
                    "N": [N],
                    "best_loc_rad": [best_loc_rad],
                    "best_infl": [best_infl],
                }

                # Add metrics data to dictionary dynamically
                for i, metric in enumerate(metric_names):
                    df_dict[f"{metric}"] = [result_avg[i, best_rmse_ind]]
                    df_dict[f"{metric}_dstd"] = [result_std[i, best_rmse_ind]]
                
                df = pd.DataFrame(df_dict)

                csv_file = os.path.join("save", f"benchmarks_random_h_noise_{dataset}.csv")
                
                if not trial_processed:
                    csv_file = os.path.join("save", f"benchmarks_random_h_noise_{dataset}.csv")

                    directory = os.path.dirname(csv_file)
                    if not os.path.exists(directory):
                        os.makedirs(directory)  # Create the directory if it doesn't exist

                    if os.path.isfile(csv_file):
                        df.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(csv_file, mode='w', index=False)

                # print("Results saved to:", csv_file)

if __name__ == "__main__":
    main(GRID_SEARCH_INFO)