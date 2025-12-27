from typing import Set
import numpy as np
import time
import os

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from utils import L63, L96, rk4, etd_rk4_wrapper
from utils import AverageMeter, mystery_operator, get_mean_std
from visualization import plot_particle_trajectories_with_histograms
from EnKF_utils import loc_EnKF_analysis, EnKF_analysis, post_process, mean0
from localization import dist2coeff, create_loc_mat
from loss import compute_loss, compute_es, compute_mean_pen, compute_cov_pen
from networks import NaiveNetwork, SetTransformer, Simple_MLP, Upsampling_MLP
from utils import get_dataloader, partial_obs_operator, setup_optimizer_and_scheduler, load_checkpoint
from training_utils import set_models, test_model
from config.cli import get_parameters
from utils import redirect_output
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

    # method = "LETKF"
    method = args.v
    if method == 'EnKF':
        method = 'EnKF_PertObs'
    elif method == 'iEnKS-PertObs':
        method = 'iEnKF_PertObs'
    elif method == 'ESRF':
        method = 'LETKF'
    elif method == 'LETKF':
        method = 'LETKF'
    method_data = df[(df['method'] == method) & (df['N'] == args.N)]
    
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


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='Test RRMSE vs Noise')
    # parser.add_argument('--plot', action='store_true', help='Plot the results')
    args = get_parameters()
    enkf_args = get_parameters()
    # os.makedirs('results', exist_ok=True)

    # enkf_data = torch.load('results/enkf_rrmse.pt')
    # enkf_rrmse = enkf_data['rrmse']
    # noise = enkf_data['noise']

    # enkf_args.test_only = True  # Set test_only to True for analysis
    # # enkf_args.plot = args.plot
    # enkf_args.N = 10
    # enkf_args.v = 'EnKF'  # Set the method to EnKF for testing
    # enkf_args.dataset = 'lorenz63'
    # enkf_args.cp_load_path = 'no'  # No checkpoint loading for this test
    # enkf_args.seed = 42
    # enkf_args.random_noise = True  # Set access to noise for EnKF
    # enkf_args.access_to_noise = False  # Access to noise for EnKF
    # enkf_args.test_traj_num = 64*16
    # enkf_args.test_batch_size = 64
    # enkf_args.access_to_H = False

    # # # enkf_args.no_localization = True

    # if enkf_args.N != 1000:
    #         sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(enkf_args)
    #         dapper_array = sigma_y_1_array
    #         loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
    #         print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
    #         print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
    #         print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
    # if enkf_args.seed is not None and enkf_args.seed != "None":
    #     torch.manual_seed(int(enkf_args.seed))
    # # args.test_only = True  # Set test_only to True for analysis
    # test_loader = get_dataloader(enkf_args, test_only=True)
    # H_info = partial_obs_operator(enkf_args.ori_dim, enkf_args.obs_inds, enkf_args.device)

    # # # # # enkf_rrmse, noise = test_ClassicFilter(
    # # # # #     test_loader, enkf_args, access_to_noise=True, plot=True, H_info=H_info,
    # # # # #     plot_figures=False, loc_radius=loc_radius, infl=infl, save_pdf=False
    # # # # # )
    # enkf_rrmse, noise = test_ClassicFilter_v2(test_loader, enkf_args, plot=True, H_info=H_info, plot_figures=False, 
    #             save_pdf=True, infl=infl, loc_radius=loc_radius)
    # torch.save({'rrmse': enkf_rrmse, 'noise': noise}, 'results/enkf_rrmse_l63_approx_H_and_gamma.pt')
    enkf_data = torch.load('results/enkf_rrmse_l63_approx_H_and_gamma.pt')
    enkf_rrmse = enkf_data['rrmse']
    noise = enkf_data['noise']

    # for num in enkf_rrmse:
    #     print(num)
    # 1+1


    # enkf_args.test_only = True  # Set test_only to True for analysis
    # # enkf_args.plot = args.plot
    # enkf_args.N = 10
    # enkf_args.v = 'EnKF'  # Set the method to EnKF for testing
    # enkf_args.dataset = 'lorenz96'
    # enkf_args.cp_load_path = 'no'  # No checkpoint loading for this test
    # enkf_args.seed = 42
    # enkf_args.random_noise = True  # Set access to noise for EnKF
    # enkf_args.access_to_noise = True  # Access to noise for EnKF
    # enkf_args.test_traj_num = 64*16
    # enkf_args.test_batch_size = 64

    # if enkf_args.N != 1000:
    #         sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(enkf_args)
    #         dapper_array = sigma_y_1_array
    #         loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
    #         print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
    #         print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
    #         print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
    # if enkf_args.seed is not None and enkf_args.seed != "None":
    #     torch.manual_seed(int(enkf_args.seed))
    # # args.test_only = True  # Set test_only to True for analysis
    # test_loader = get_dataloader(enkf_args, test_only=True)
    # H_info = partial_obs_operator(enkf_args.ori_dim, enkf_args.obs_inds, enkf_args.device)

    # enkf_no_noise_rrmse, noise = test_ClassicFilter(
    #     test_loader, enkf_args, access_to_noise=False, plot=True, H_info=H_info,
    #     plot_figures=False, loc_radius=loc_radius, infl=infl, save_pdf=False
    # )
    # torch.save({'rrmse': enkf_no_noise_rrmse, 'noise': noise}, 'results/enkf_no_noise_rrmse_l96.pt')

    smf_args.test_only = True  # Set test_only to True for analysis
    # enkf_args.plot = args.plot
    smf_args.N = 10
    smf_args.v = 'SMF'  # Set the method to EnKF for testing
    smf_args.dataset = 'lorenz96'
    smf_args.cp_load_path = 'no'  # No checkpoint loading for this test
    smf_args.seed = 42
    smf_args.random_noise = True  # Set access to noise for EnKF
    smf_args.access_to_noise = True  # Access to noise for EnKF
    smf_args.test_traj_num = 64*16
    smf_args.test_batch_size = 64

    if enkf_args.N != 1000:
            sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(enkf_args)
            dapper_array = sigma_y_1_array
            loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
            print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
            print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
            print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
    if enkf_args.seed is not None and enkf_args.seed != "None":
        torch.manual_seed(int(enkf_args.seed))
    # args.test_only = True  # Set test_only to True for analysis
    test_loader = get_dataloader(enkf_args, test_only=True)
    H_info = partial_obs_operator(enkf_args.ori_dim, enkf_args.obs_inds, enkf_args.device)

    enkf_no_noise_rrmse, noise = test_ClassicFilter(
        test_loader, enkf_args, access_to_noise=False, plot=True, H_info=H_info,
        plot_figures=False, loc_radius=loc_radius, infl=infl, save_pdf=False
    )
    torch.save({'rrmse': enkf_no_noise_rrmse, 'noise': noise}, 'results/enkf_no_noise_rrmse_l96.pt')
    # enkf_no_noise_data = torch.load('results/enkf_no_noise_rrmse_l96.pt')
    # enkf_no_noise_rrmse = enkf_no_noise_data['rrmse']
    # enkf_no_noise_noise = enkf_no_noise_data['noise']

    # esrf_data = torch.load('results/esrf_rrmse.pt')
    # esrf_rrmse = esrf_data['rrmse']
    # esrf_noise = esrf_data['noise']

    # if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='Test RRMSE vs Noise')
    # parser.add_argument('--plot', action='store_true', help='Plot the results')
    args = get_parameters()
    ienkf_args = get_parameters()
    # os.makedirs('results', exist_ok=True)

    # # enkf_data = torch.load('results/enkf_rrmse.pt')
    # # enkf_rrmse = enkf_data['rrmse']
    # # noise = enkf_data['noise']

    # ienkf_args.test_only = True  # Set test_only to True for analysis
    # # enkf_args.plot = args.plot
    # ienkf_args.N = 10
    # ienkf_args.v = 'iEnKS-PertObs'  # Set the method to EnKF for testing
    # ienkf_args.dataset = 'lorenz63'
    # ienkf_args.cp_load_path = 'no'  # No checkpoint loading for this test
    # ienkf_args.seed = 10
    # ienkf_args.random_noise = True  # Set access to noise for EnKF
    # ienkf_args.access_to_noise = False  # Access to noise for EnKF
    # ienkf_args.test_traj_num = 64*16
    # ienkf_args.test_batch_size = 64
    # ienkf_args.access_to_H = False
    # # # ienkf_args.no_localization = True

    # if ienkf_args.N != 1000:
    #         sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(ienkf_args)
    #         dapper_array = sigma_y_1_array
    #         loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
    #         print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
    #         print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
    #         print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
    # if ienkf_args.seed is not None and ienkf_args.seed != "None":
    #     torch.manual_seed(int(ienkf_args.seed))
    # # # args.test_only = True  # Set test_only to True for analysis
    # test_loader = get_dataloader(ienkf_args, test_only=True)
    # H_info = partial_obs_operator(ienkf_args.ori_dim, ienkf_args.obs_inds, ienkf_args.device)

    # ienkf_rrmse, noise = test_ClassicFilter_v2(
    #     test_loader, ienkf_args, plot=True, H_info=H_info,
    #     plot_figures=False, loc_radius=loc_radius, infl=infl, save_pdf=False
    # )
    # torch.save({'rrmse': ienkf_rrmse, 'noise': noise}, 'results/ienkf_rrmse_l63_approx_H_and_gamma.pt')
    ienkf_data = torch.load('results/ienkf_rrmse_l63_approx_H_and_gamma.pt')
    ienkf_rrmse = ienkf_data['rrmse']
    ienkf_noise = ienkf_data['noise']

    # esrf_args = get_parameters()
    # esrf_args.test_only = True  # Set test_only to True for analysis
    # # enkf_args.plot = args.plot
    # esrf_args.N = 10
    # esrf_args.v = 'ESRF'  # Set the method to EnKF for testing
    # esrf_args.dataset = 'lorenz96'
    # esrf_args.cp_load_path = 'no'  # No checkpoint loading for this test
    # esrf_args.seed = 42
    # esrf_args.random_noise = True  # Set access to noise for EnKF
    # esrf_args.access_to_noise = False  # Access to noise for EnKF
    # esrf_args.test_traj_num = 64*16
    # esrf_args.test_batch_size = 64
    # esrf_args.access_to_H = False
    # # # esrf_args.no_localization = True

    # if esrf_args.N != 1000:
    #         sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(esrf_args)
    #         dapper_array = sigma_y_1_array
    #         loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
    #         print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
    #         print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
    #         print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
    # if esrf_args.seed is not None and esrf_args.seed != "None":
    #     torch.manual_seed(int(esrf_args.seed))
    # # args.test_only = True  # Set test_only to True for analysis
    # test_loader = get_dataloader(esrf_args, test_only=True)
    # H_info = partial_obs_operator(esrf_args.ori_dim, esrf_args.obs_inds, esrf_args.device)
    # # infl = 1.05
    # esrf_rrmse, noise = test_ClassicFilter_v2(test_loader, esrf_args, plot=True, H_info=H_info, plot_figures=False,
    #             save_pdf=True, infl=infl, loc_radius=loc_radius)

    # torch.save({'rrmse': esrf_rrmse, 'noise': noise}, 'results/esrf_rrmse_l96_approx_H_and_gamma.pt')
    # esrf_data = torch.load('results/esrf_rrmse_l63_approx_gamma.pt')
    # esrf_rrmse = esrf_data['rrmse']
    # esrf_noise = esrf_data['noise']

    # letkf_data = torch.load('results/letkf_rrmse_l96_approx_H_and_gamma.pt')
    # letkf_rrmse = letkf_data['rrmse']
    # letkf_noise = letkf_data['noise']

    letkf_args = get_parameters()
    # letkf_args.test_only = True  # Set test_only to True for analysis
    # # letkf_args.plot = args.plot
    # letkf_args.N = 10
    # letkf_args.v = 'LETKF'  # Set the method to LETKF for testing
    # letkf_args.dataset = 'lorenz96'
    # letkf_args.cp_load_path = 'no'  # No checkpoint loading for this test
    # letkf_args.seed = 42
    # letkf_args.random_noise = True  # Set access to noise for LETKF
    # letkf_args.access_to_noise = False  # Access to noise for LETKF
    # letkf_args.test_traj_num = 64*16
    # letkf_args.test_batch_size = 64
    # letkf_args.access_to_H = False

    # if letkf_args.N != 1000:
    #         sigma_y_1_array, sigma_y_0_7_array = get_benchmarks(letkf_args)
    #         dapper_array = sigma_y_1_array
    #         loc_radius, infl, rmse_dapper, rrmse_dapper = dapper_array[0,0], dapper_array[0,1], dapper_array[0,2], dapper_array[0,3]
    #         print(f"RMSE from DAPPER: {rmse_dapper:.3f}.")
    #         print(f"RRMSE from DAPPER: {rrmse_dapper:.3f}.")
    #         print(f"Inflation: {infl}; Localization Radius: {loc_radius}")
    # if letkf_args.seed is not None and letkf_args.seed != "None":
    #     torch.manual_seed(int(letkf_args.seed))
    # # args.test_only = True  # Set test_only to True for analysis
    # test_loader = get_dataloader(letkf_args, test_only=True)
    # H_info = partial_obs_operator(letkf_args.ori_dim, letkf_args.obs_inds, letkf_args.device)

    # letkf_rrmse, noise = test_ClassicFilter_v2(test_loader, letkf_args, plot=True, H_info=H_info, plot_figures=False,
    #             save_pdf=True, infl=infl, loc_radius=loc_radius)
    # # # letkf_rrmse, noise = test_ClassicFilter(
    # # #     test_loader, letkf_args, access_to_noise=True, plot=True, H_info=H_info,
    # # #     plot_figures=False, loc_radius=loc_radius, infl=infl, save_pdf=False
    # # # )
    # torch.save({'rrmse': letkf_rrmse, 'noise': noise}, 'results/letkf_rrmse_l96_approx_H_and_gamma.pt')
    # for num in letkf_rrmse:
    #     print(num)

    model_args = args
    model_args.test_only = True  # Set test_only to True for analysis
    # # # model_args.plot = args.plot
    model_args.N = 10
    model_args.v = 'Affine-ydagger'
    model_args.dataset = 'lorenz63'
    # # model_args.cp_load_path = 'save/2025-07-24_14-27lorenz96_1.0_10_60_8192_nl2_joint_Affine-ydagger/cp_1000.pth'
    # # # # # model_args.cp_load_path = 'save/2025-08-04_15-29lorenz96_None_10_60_8192_nl2_joint_Affine-ydagger/cp_1000.pth'
    # # # # # model_args.cp_load_path = 'save/2025-08-04_15-29lorenz63_None_10_60_8192_nl2_joint_Affine-ydagger/cp_1000.pth'
    # # # # # model_args.cp_load_path = 'save/2025-08-04_23-42lorenz96_None_10_60_8192_nl2_joint_Affine-ydagger_new_08041529_l96/cp_1900.pth'
    # # # # # model_args.cp_load_path = 'save/2025-08-04_23-43lorenz63_None_10_60_8192_nl2_joint_Affine-ydagger_new_08041529_l63/cp_2000.pth'
    # # # # model_args.cp_load_path = 'save/2025-08-05_11-41lorenz63_None_10_60_8192_nl2_joint_Affine-ydagger_2k_1k_l63_tuned/cp_1000.pth'
    # model_args.cp_load_path = 'save/2025-08-05_11-42lorenz96_None_10_60_8192_nl2_joint_Affine-ydagger_2k_1k_l96_tuned/cp_1000.pth'
    model_args.cp_load_path = 'save/2025-08-08_20-06lorenz63_None_10_60_8192_nl2_joint_Affine-ydagger_3k_sigma^2_l63/cp_3000.pth'
    # model_args.cp_load_path = 'save/2025-08-08_20-07lorenz96_None_10_60_8192_nl2_joint_Affine-ydagger_3k_sigma^2_l96/cp_3000.pth'
    # model_args.seed = 42
    # model_args.random_noise = True  # Set access to noise for model
    # model_args.no_localization = True
    # model_args.test_traj_num = 64*16
    # model_args.test_batch_size = 64
    # model_args.use_data_parallel = False
    # if model_args.seed is not None and model_args.seed != "None":
    #     torch.manual_seed(int(model_args.seed))
    # args.test_only = True  # Set test_only to True for analysis
    # test_loader = get_dataloader(model_args, test_only=True)
    # model_list = set_models(model_args)
    # H_info = partial_obs_operator(model_args.ori_dim, model_args.obs_inds, model_args.device)

    # if model_args.v == 'Affine-ydagger':
    #     model_Avhat, model_Ayhat, model_Aydag, infl_model, local_model, st_model1, st_model2 = model_list
    # else:
    #     model, infl_model, local_model, st_model1, st_model2 = model_list

    # # # optimizer
    # optimizer, scheduler = setup_optimizer_and_scheduler(model_list, model_args)
    folder_name = model_args.cp_load_path.split('/')[1]
    import os
    output_dir = os.path.join("testing", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    # if model_args.cp_load_path != "no":
    #     load_checkpoint(model_list, None, None, filename=model_args.cp_load_path, use_data_parallel=False)
    #     # for name, net in zip(
    #     #     ['model_Avhat','model_Ayhat','model_Aydag','infl_model','local_model','st_model1','st_model2'],
    #     #     model_list
    #     # ):
    #     #     net.eval()
    #     # print("Test Only")
    # model_rrmse, model_noise = \
    #         test_model(test_loader, model_list, model_args, H_info=H_info, plot_figures=False, fig_name=f'testing/{folder_name}/test_only_{args.N}', analysis = False, plot=True)
    # mean_rmse, std_rmse, mean_rmv, std_rmv, mean_rrmse, std_rrmse, mean_crps, std_crps, no_nan_percent, loc_tensor = \
    #         test_model(test_loader, model_list, model_args, H_info=H_info, plot_figures=False, fig_name=f'testing/{folder_name}/test_only_{args.N}', analysis = False, plot=False)
    # print(f"RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    # print(f"RRMSE: {mean_rrmse:.3f} ± {std_rrmse:.3f}")
    # print(f"RMV: {mean_rmv:.3f} ± {std_rmv:.3f}")
    # print(f"CRPS: {mean_crps:.3f} ± {std_crps:.3f}")
    # torch.save({'rrmse': model_rrmse, 'noise': noise}, 'results/model_rrmse_l96_3k_sigma^2.pt')
    model_data = torch.load('results/model_rrmse_l63_3k_sigma^2.pt')
    model_rrmse = model_data['rrmse']
    model_noise = model_data['noise']
    #plot enkf_rrmse and model_rrmse as scatter plot
    plt.figure(figsize=(10, 6))
    # cat model_noise to itself 4 times
    # model_noise = model_noise.repeat(4)
    # enkf_noise = noise.repeat(4)
    # form a size 64 vector from model_rrmse and enkf_rrmse by averaging every 4 elements
    #increase the fontsize of legend, title, labels, and ticks
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    # for i in range(ienkf_rrmse.shape[0]):
    #     print(ienkf_rrmse[i])
    _model_rrmse = model_rrmse.view(16, 64).mean(dim=0)
    # _ienkf_rrmse = ienkf_rrmse.view(16, 64).mean(dim=0)
    _ienkf_rrmse = torch.nanmean(ienkf_rrmse.view(16, 64), dim=0)
    _enkf_rrmse = enkf_rrmse.view(16, 64).mean(dim=0)
    # _no_noise_enkf_rrmse = enkf_no_noise_rrmse.view(16, 64).mean(dim=0)
    # _esrf_rrmse = esrf_rrmse.view(16, 64).mean(dim=0)
    # _esrf_rrmse = esrf_rrmse.view(16, 64).mean(dim=0)
    # _letkf_rrmse = letkf_rrmse.view(16, 64).mean(dim=0)
    plt.plot(model_noise.cpu().numpy(), _model_rrmse.cpu().numpy(), label='Ours', color = 'blue')
    plt.plot(noise.cpu().numpy(), _enkf_rrmse.cpu().numpy(), label='EnKF', color='red')
    # plt.plot(noise.cpu().numpy(), _esrf_rrmse.cpu().numpy(), label='ESRF', color='green')
    plt.plot(noise.cpu().numpy(), _ienkf_rrmse.cpu().numpy(), label='iEnKF', color='black')
    # plt.plot(noise.cpu().numpy(), _letkf_rrmse.cpu().numpy(), label='LETKF', color='black')
    # plt.plot(noise.cpu().numpy(), _no_noise_enkf_rrmse.cpu().numpy(), label='EnKF (No Noise Access)', color='purple')
    # use bbox to make legend outside the plot to the right
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # leave 20% space on right
    plt.legend()
    plt.title('RRMSE vs Noise, Lorenz 63 Model')
    #convert the sigma_y to latex format
    plt.xlabel('Obs Noise, $\sigma_y$')
    plt.ylabel('RRMSE')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rrmse_vs_noise.png'))
    plt.close()

    #make a plot showing the success rate of each method at each noise level, where success
    # is number of non nan rrmse values / 16 for each noise level
    success_rate_ienkf = []
    for i in range(64):
        count_non_nan = torch.sum(~torch.isnan(ienkf_rrmse.view(16, 64)[:, i])).item()
        success_rate_ienkf.append(count_non_nan / 16)
    success_rate_ienkf = np.array(success_rate_ienkf)

    # success_rate_letkf = []
    # for i in range(64):
    #     count_non_nan = torch.sum(~torch.isnan(letkf_rrmse.view(16, 64)[:, i])).item()
    #     success_rate_letkf.append(count_non_nan / 16)
    # success_rate_letkf = np.array(success_rate_letkf)

    success_rate_model = []
    for i in range(64):
        count_non_nan = torch.sum(~torch.isnan(model_rrmse.view(16, 64)[:, i])).item()
        success_rate_model.append(count_non_nan / 16)
    success_rate_model = np.array(success_rate_model)

    success_rate_enkf = []
    for i in range(64):
        count_non_nan = torch.sum(~torch.isnan(enkf_rrmse.view(16, 64)[:, i])).item()
        success_rate_enkf.append(count_non_nan / 16)
    success_rate_enkf = np.array(success_rate_enkf)

    #now make a plot of the success rate of each method at each noise level for enkf, model, letkf
    #make a line plot, not a bar chart
    plt.figure(figsize=(10, 6))
    # print(success_rate_enkf, success_rate_model, success_rate_letkf)
    # noise should be an array of length 64, e.g. np.linspace(0.1, 1.0, 64)
    plt.plot(noise.cpu().numpy(), success_rate_model*100, label='Ours', color='black', marker='s', markersize=4)
    plt.plot(noise.cpu().numpy(), success_rate_enkf*100,  label='EnKF', color='blue', marker='o', markersize=4)
    # plt.plot(noise.cpu().numpy(), success_rate_letkf*100, label='LETKF', color='purple', marker='^', markersize=4)
    plt.plot(noise.cpu().numpy(), success_rate_ienkf*100, label='iEnKF', color='red', marker='x', markersize=4)

    # Reference line at 100%
    plt.axhline(100, color='gray', linestyle='--', label='Reference (100%)')

    plt.xlabel('Observation Noise Level')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate vs Observation Noise')
    plt.ylim(50, 105)  # show dips clearly but keep 100% visible
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'success_rate_vs_noise.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    # cat model_noise to itself 4 times
    model_noise = model_noise.repeat(16)
    enkf_noise = noise.repeat(16)
    # esrf_noise = esrf_noise.repeat(16)
    # letkf_noise = letkf_noise.repeat(16)
    # ienkf_noise = ienkf_noise.repeat(16)
    # _no_noise_enkf_rrmse = _no_noise_enkf_rrmse.repeat(16)
    # no_noise_enkf_rrmse = enkf_no_noise_rrmse.repeat(16)
    # enkf_rrmse = enkf_rrmse.view(4, 64).mean(dim=0)
    # esrf_rrmse = esrf_rrmse.view(4, 64).mean(dim=0)
    # letkf_rrmse = letkf_rrmse.view(4, 64).mean(dim=0)
    # form a size 64 vector from model_rrmse and enkf_rrmse by averaging every 4 elements
    # model_rrmse = model_rrmse.view(4, 64).mean(dim=0)
    # enkf_rrmse = enkf_rrmse.view(4, 64).mean(dim=0)
    plt.scatter(model_noise.cpu().numpy(), model_rrmse.cpu().numpy(), label='Model RRMSE', color = 'blue')
    plt.scatter(x=enkf_noise.cpu().numpy(), y=enkf_rrmse.cpu().numpy(), label='EnKF RRMSE', color='red')
    # plt.scatter(x=esrf_noise.cpu().numpy(), y=esrf_rrmse.cpu().numpy(), label='ESRF RRMSE', color='green')
    # plt.scatter(x=letkf_noise.cpu().numpy(), y=letkf_rrmse.cpu().numpy(), label='LETKF RRMSE', color='purple')
    # plt.scatter(x=letkf_noise.cpu().numpy(), y=_no_noise_enkf_rrmse.cpu().numpy(), label='EnKF (No Noise Access) RRMSE', color='black')
    plt.legend()
    plt.title('RRMSE vs Noise, Lorenz 96 Model')
    plt.xlabel('Obs Noise, sigma_y')
    plt.ylabel('RRMSE')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rrmse_vs_noise_unmean.png'))
    plt.close()