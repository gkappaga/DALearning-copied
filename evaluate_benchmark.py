import os
import os, torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x


import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # may throw if an op has no det. path

# Turn off TF32 to avoid GEMM path changes
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# CUBLAS determinism for matmuls (set before import torch if possible)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8"
import torch.nn as nn
import pandas as pd
import numpy as np

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from train_test_utils import test_ClassicFilter
from train_test_utils_v2 import test_ClassicFilter_v2

def get_benchmarks(args):
    """
    This function processes benchmark data from a CSV file, splitting it by `sigma_y` values 1 and 0.7,
    and extracting specific columns for each `method`, then combining them into a 2*N*5 numpy array.
    
    Args:
        args: An object or namespace with a `dataset` attribute specifying the dataset name.
        
    Returns:
        result_dict: A dictionary where each key is a method, and the value is a 2*N*5 numpy array.
    """
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
    df = pd.read_csv(file_path, usecols=['method', 'N', 'sigma_y', 'best_loc_rad','best_infl','rmse', 'rrmse_mean'])

    method = args.v
    if method == 'SMF' or method == 'Affine-ydagger' or method == 'EnKF':
        method = 'EnKF_PertObs'
    if method == 'ESRF' or method == 'LETKF':
        method = 'EnKF_PertObs'
    if method == 'iEnKS-PertObs':
        method = 'iEnKF_PertObs'
    if method == 'iEnKS-Sqrt':
        method = 'iEnKF'
    # if method == 'LETKF':
    # method = 'LETKF'
    
    # method = args.v
    if method == 'iEnKS-PertObs':
        method = 'iEnKF_PertObs'
    method_data = df[(df['method'] == method) & (df['N'] == 10)]
    
    # Filter rows where sigma_y == 1 and 0.7
    sigma_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad','best_infl','rmse', 'rrmse_mean']]
    
    # Convert to numpy arrays
    sigma_y_1_array = sigma_y_1.to_numpy()
    print(sigma_y_1_array)
    if(np.isnan(sigma_y_1_array[0][0])):
        sigma_y_1_array[0][0] = 4
    sigma_y_0_7_array = sigma_y_0_7.to_numpy()

    return sigma_y_1_array, sigma_y_0_7_array

import matplotlib.pyplot as plt

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


def sweep_ensemble_sizes_for_rrmse(method_name, args_template):
    """
    Sweep over ensemble sizes N_list, evaluate SMF (or other method),
    and produce a plot of mean RRMSE vs N.
    """
    N_list = [5, 10, 15, 20, 40, 60, 100]
    rrmse_means = []

    print("\n=== Running Ensemble Sweep ===\n")

    for N in N_list:
        print(f"\n--- Evaluating N = {N} ---")

        # Copy args and override N
        args = get_parameters()
        args.N = N
        args.v = method_name          # e.g. 'SMF'
        args.test_only = True
        args.seed = 42
        args.random_noise = False     # IMPORTANT for consistent SMF
        args.access_to_noise = False
        args.redirect_output = False  # no file redirects
        args.test_batch_size = 64
        args.test_traj_num = 64*16     # can reduce for speed
        args.dataset = 'lorenz63'

        # H operator
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)
        test_loader = get_dataloader(args, test_only=True)

        # Run test_ClassicFilter_v2 (automatically dispatches SMF)
        metrics_or_rrmse = test_ClassicFilter_v2(
            test_loader, args,
            plot=False, H_info=H_info,
            plot_figures=False, fig_name='tmp', save_pdf=False,
            infl=0.0, loc_radius=None
        )

        # test_ClassicFilter_v2 returns RRMSE directly if plot=False
        rrmse_mean = float(metrics_or_rrmse)
        rrmse_means.append(rrmse_mean)

        print(f"N={N}: mean RRMSE = {rrmse_mean:.4f}")

    # Create plot
    plt.figure(figsize=(6,4))
    plt.plot(N_list, rrmse_means, marker='o')
    plt.title(f"Mean RRMSE vs Ensemble Size for {method_name}")
    plt.xlabel("Ensemble Size N")
    plt.ylabel("Mean RRMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"rrmse_vs_N_{method_name}.pdf")
    plt.savefig(f"rrmse_vs_N_{method_name}.png")
    plt.show()

    return N_list, rrmse_means


if __name__ == "__main__":
    args = get_parameters()
    args = get_parameters()
    args.test_only = True  # Set test_only to True for analysis
    # enkf_args.plot = args.plot
    args.N = 20
    args.v = 'ESRF'  # Set the method to EnKF for testing
    args.dataset = 'lorenz96'
    args.cp_load_path = 'no'  # No checkpoint loading for this test
    args.seed = 42
    args.random_noise = True  # Set access to noise for EnKF
    args.access_to_noise = True  # Access to noise for EnKF
    args.test_traj_num = 64
    args.test_batch_size = 64
    args.access_to_H = True
    args.random_h = True
    suffix = ""
    
    if args.v == "EnKF" and hasattr(args, "access_to_noise"):
        suffix = f"_access_{args.access_to_noise}"

    folder_name = os.path.join('save/benchmark_models/', f"benchmark_{args.dataset}_varyingnoise_{args.v}{suffix}_{args.N}")
    # folder_name = os.path.join('save/benchmark_models/', f"benchmark_{args.dataset}_varyingnoise_{args.v}_{args.N}")
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    
    # redirect output
    print_run_signature('eval', args)
    with redirect_output(folder_name, filename="test_output.txt", enable_redirect=args.redirect_output):
        print(f'{args.access_to_noise}')
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # modify test_batch_size
        if args.N == 100:
            args.test_batch_size = args.test_batch_size // 2
        test_loader = get_dataloader(args, test_only=True)
        
        # print test information
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.")
    
        # get optimal parameters
        if args.N != 1000:
            sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(args)
            dapper_array = sigma_y_1_array
            # if args.sigma_y == 1:
            #     dapper_array = sigma_y_1_array
            # elif args.sigma_y == 0.7:
            #     dapper_array = sigma_y_0_7_array
            # else:
            #     raise NotImplementedError
            loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
            print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
            print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
            print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
        
        
        # test
        print(f"Test {args.v} Results")
        loss_list_nn = []
        # mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn = \
        #     test_ClassicFilter(test_loader, args, plot=False, H_info=H_info, plot_figures=False, fig_name=f'{folder_name}/test_{args.N}', save_pdf=True, access_to_noise=True, infl=infl, loc_radius=loc_radius)
        rmse = test_ClassicFilter_v2(test_loader, args, plot=False, H_info=H_info, plot_figures=False, fig_name=f'{folder_name}/test_{args.N}', save_pdf=True, infl=1.1, loc_radius=None)
        # print(f"RMSE: {mean_rmse_nn:.3f} ± {std_rmse_nn:.3f}")
        # print(f"RRMSE: {mean_rrmse_nn:.3f} ± {std_rrmse_nn:.3f}")
        # print(f"RMV: {mean_rmv_nn:.3f} ± {std_rmv_nn:.3f}")
        # print(f"CRPS: {mean_crps_nn:.3f} ± {std_crps_nn:.3f}")
        # print(f'No NAN Percentage: {no_nan_percent_nn * 100: .2f}%')
        print(rmse)

        
            
        # save results
        # tensor_dict = {
        #     'nn':{
        #         'mean_rmse':mean_rmse_nn,
        #         'std_rmse':std_rmse_nn,
        #         'mean_rrmse':mean_rrmse_nn,
        #         'std_rrmse':std_rrmse_nn,
        #         'mean_rmv':mean_rmv_nn,
        #         'std_rmv':std_rmv_nn,
        #         'valid_percent':no_nan_percent_nn,
        #         'loc_diff_dist':args.diff_dist,
        #     },
        #     'cp_load_path': args.cp_load_path,
        #     'sigma_y': args.sigma_y,
        # }
        
        # print(torch.mean((ens_tensor_enkf.mean(dim=2) - ens_tensor_nn.mean(dim=2))**2, dim=(1,2))[:100])
        
        # record_name = f"output_records_{args.N}.pt"
        # record_path = os.path.join(folder_name, record_name)
        # torch.save(tensor_dict, record_path)
        # if args.cp_load_path != "no":
        #     if args.zero_infl:
        #         torch.save(tensor_dict, os.path.join(folder_name, f"output_records_zero_infl_{args.N}.pt"))
        #     else:
        #         torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))