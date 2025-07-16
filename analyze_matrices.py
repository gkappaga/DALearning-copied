from typing import Set
import numpy as np
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
    mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn, loc_tensor, result, norm = \
            test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'testing/test_only_{args.N}', analysis = True)
    rmse_list.append(mean_rmse_nn)
    rrmse_list.append(mean_rrmse_nn)
    print("Average RMSE:", torch.mean(torch.tensor(rmse_list)))
    print("Average R-RMSE:", torch.mean(torch.tensor(rrmse_list)))
    return result, norm

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
    print(args.cp_load_path)
    if args.cp_load_path != "no":
        load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)
        for name, net in zip(
            ['model_Avhat','model_Ayhat','model_Aydag','infl_model','local_model','st_model1','st_model2'],
            model_list
        ):
            net.eval()
        print("Test Only")
    result, norm = run_analyses(test_loader, args, H_info=H_info, plot_figures=True, fig_name=f'testing/test_only_{args.N}')
    sep = torch.mean(result, dim = 1)
    sep_n = torch.mean(norm, dim = 1)
    plt.plot(sep[:, 0].cpu().numpy(), label='yhat')
    plt.plot(sep_n[:, 0].cpu().numpy(), label='yhat_norm')
    plt.xlabel('Timestep')
    plt.ylabel('Difference')
    plt.title('Difference between Ayhat and K')
    plt.legend()
    plt.savefig(f'testing/Ayhat_diff_{args.N}.png')
    plt.close()
    plt.plot(sep[:, 1].cpu().numpy(), label='ydag')
    plt.plot(sep_n[:, 1].cpu().numpy(), label='ydag_norm')
    plt.xlabel('Timestep')
    plt.ylabel('Difference')
    plt.title('Difference between Ayhat and K')
    plt.legend()
    plt.savefig(f'testing/Aydag_diff_{args.N}.png')
    plt.close()
    plt.plot(sep[:, 2].cpu().numpy(), label='Avhat')
    plt.plot(sep_n[:, 2].cpu().numpy(), label='Avhat_norm')
    plt.xlabel('Timestep')
    plt.ylabel('Difference')
    plt.title('Difference between Avhat and I')
    plt.legend()
    plt.savefig(f'testing/Avhat_diff_{args.N}.png')
    plt.close()
    traj = result[:, 0, :].cpu().numpy()
    plt.plot(traj[:, 0], label='yhat')
    plt.plot(traj[:, 1], label='ydag')
    plt.plot(traj[:, 2], label='Avhat')
    plt.xlabel('Timestep')
    plt.ylabel('Difference')
    plt.title('Differences over time')
    plt.legend()
    plt.savefig(f'testing/differences_{args.N}.png')
    plt.close()

                