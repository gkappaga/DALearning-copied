from typing import Set
import numpy as np
import time

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

def run_analyses(loader, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig'):
    rmse_list, rrmse_list = [], []
    mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn, loc_tensor, result, norm, an_results, aydag, ayhat, kalmans, avhat = \
            test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'testing/{folder_name}/test_only_{args.N}', analysis = True)
    rmse_list.append(mean_rmse_nn)
    rrmse_list.append(mean_rrmse_nn)
    print("Average RMSE:", torch.mean(torch.tensor(rmse_list)))
    print("Average R-RMSE:", torch.mean(torch.tensor(rrmse_list)))
    return result, norm, an_results, aydag, ayhat, kalmans, avhat

if __name__ == "__main__":
    args = get_parameters()
    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))
    # args.test_only = True  # Set test_only to True for analysis
    test_loader = get_dataloader(args, test_only=True)
    model_list = set_models(args)
    H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

    if args.v == 'Affine-ydagger':
        model_Avhat, model_Ayhat, model_Aydag, infl_model, local_model, st_model1, st_model2 = model_list
    else:
        model, infl_model, local_model, st_model1, st_model2 = model_list

    # optimizer
    optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)
    folder_name = args.cp_load_path.split('/')[1]
    print(folder_name)
    print(args.cp_load_path)
    import os
    output_dir = os.path.join("testing", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.cp_load_path != "no":
        load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)
        for name, net in zip(
            ['model_Avhat','model_Ayhat','model_Aydag','infl_model','local_model','st_model1','st_model2'],
            model_list
        ):
            net.eval()
        print("Test Only")
    result, norm, an_results, aydag, ayhat, kalmans, avhat = run_analyses(test_loader, args, H_info=H_info, plot_figures=True, fig_name=f'testing/{folder_name}/test_only_{args.N}')
    sep = torch.mean(result, dim = 1)
    sep_n = torch.mean(norm, dim = 1)
    sep_std = torch.std(result, dim = 1)
    sep_n_std = torch.std(norm, dim = 1)
    labels = ['Ayhat', 'Aydag', 'Avhat']

    for idx, name in enumerate(labels):
        x      = np.arange(sep.shape[0])
        m1     = sep[:, idx].cpu().numpy()
        e1     = sep_std[:, idx].cpu().numpy()
        m2     = sep_n[:, idx].cpu().numpy()
        e2     = sep_n_std[:, idx].cpu().numpy()

        plt.figure()
        # first curve with errorbars
        plt.plot(x, m1, label=f'{name}',      linewidth=2)
        plt.plot(x, m2, label=f'{name}_norm', linewidth=2)

        # shade ±1 std
        plt.fill_between(x, m1-2*e1, m1+2*e1, alpha=0.5)
        plt.fill_between(x, m2-2*e2, m2+2*e2, alpha=0.5)

        plt.xlabel('Timestep')
        plt.ylabel('Difference')
        if name == 'Avhat':
            plt.title(f'Difference between {name} and I')
        elif name == 'Ayhat':
            plt.title(f'Difference between {name} and -K')
        else:
            plt.title(f'Difference between {name} and K')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'testing/{folder_name}/{name}_diff_{args.N}.png')
        plt.close()
    traj = result[:, 0, :].cpu().numpy()
    plt.plot(traj[:, 0], label='yhat')
    plt.plot(traj[:, 1], label='ydag')
    plt.plot(traj[:, 2], label='Avhat')
    plt.xlabel('Timestep')
    plt.ylabel('Difference')
    plt.title('Differences over time')
    plt.legend()
    plt.savefig(f'testing/{folder_name}/differences_{args.N}.png')
    plt.close()

    x      = np.arange(sep.shape[0])
    m1     = sep[:, 3].cpu().numpy()
    e1     = sep_std[:, 3].cpu().numpy()
    m2     = sep_n[:, 3].cpu().numpy()
    e2     = sep_n_std[:, 3].cpu().numpy()
    plt.figure()
    # first curve with errorbars
    plt.plot(x, m1, label='mean of F-norm', linewidth=2)
    # shade ±1 std
    plt.fill_between(x, m1-2*e1, m1+2*e1, alpha=0.5)
    plt.xlabel('Timestep')
    plt.ylabel('Difference')
    plt.title('Difference between Aydag and Ayhat')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/aydag_ayhat_diff_{args.N}.png')
    plt.close()

    means_per_traj = an_results[0]
    stds_per_traj = an_results[1]
    plt.figure()
    # make the std the errorbar
    plt.scatter(range(1, len(means_per_traj) + 1), means_per_traj, label='Mean per Trajectory', color='blue')
    plt.errorbar(range(1, len(means_per_traj) + 1), means_per_traj, yerr=stds_per_traj, fmt='o', color='blue', capsize=5)
    plt.xlabel('Trajectory Index')
    plt.ylabel('Mean Value')
    plt.title('Mean and Std per Trajectory')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/mean_std_per_traj_{args.N}.png')
    plt.close()
    
    #heatmap the average 40 x 10 aydag, mean over first two dimensions and then heatmap the result
    aydag_mean = torch.mean(aydag, dim=(0, 1)).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(aydag_mean, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Average Aydag Matrix')
    plt.xlabel('Observation Dimension')
    plt.ylabel('State Dimension')
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/aydag_mean_{args.N}.png')
    plt.close()

    #heatmap the average 40 x 10 ayhat, mean over first two dimensions and then heatmap the result
    ayhat_mean = torch.mean(ayhat, dim=(0, 1)).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(ayhat_mean, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Average Ayhat Matrix')
    plt.xlabel('Observation Dimension')
    plt.ylabel('State Dimension')
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/ayhat_mean_{args.N}.png')
    plt.close()

    #plot the average kalman
    kalmans_mean = torch.mean(kalmans, dim=(0, 1)).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(kalmans_mean, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Average Kalman Gain Matrix')
    plt.xlabel('Observation Dimension')
    plt.ylabel('State Dimension')
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/kalman_mean_{args.N}.png')
    plt.close()

    #plot a sample kalman gain
    sample_kalman = kalmans[0, 0, :, :].cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_kalman, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Sample Kalman Gain Matrix')
    plt.xlabel('Observation Dimension')
    plt.ylabel('State Dimension')
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/sample_kalman_{args.N}.png')
    plt.close()

    #plot avhat
    avhat_mean = torch.mean(avhat, dim=(0, 1)).cpu().numpy() + torch.eye(avhat.shape[2]).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(avhat_mean, cmap='viridis', cbar=True)
    plt.title('Average Avhat Matrix')
    plt.xlabel('Observation Dimension')
    plt.ylabel('State Dimension')
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/avhat_mean_{args.N}.png')
    plt.close()

    diff_mean = torch.mean(aydag - ayhat, dim=(0, 1)).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_mean, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Difference between Aydag and Ayhat')
    plt.xlabel('Observation Dimension')
    plt.ylabel('State Dimension')
    plt.tight_layout()
    plt.savefig(f'testing/{folder_name}/aydag_ayhat_diff_mean_{args.N}.png')
    plt.close()