import os

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from training_utils import test_model, set_models


if __name__ == "__main__":
    args = get_parameters()
    
    if args.cp_load_path == "no":
        raise ValueError("The parameter cp_load_path is invalid.")
    folder_name = os.path.dirname(args.cp_load_path)
    
    # redirect output
    with redirect_output(folder_name, filename="test_output.txt"):
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
        
        # EnKF
        # print("Original EnKF with localization")

        # mean_rmse_enkf, std_rmse_enkf, mean_rmv_enkf, std_rmv_enkf, no_nan_percent_enkf = test_SequentialEnKF(test_loader, args, infl=1.09, H_info=H_info, localization=False)
        # print(f"RMSE: {mean_rmse_enkf:.3f} ± {std_rmse_enkf:.3f}")
        # print(f"RMV: {mean_rmv_enkf:.3f} ± {std_rmv_enkf:.3f}")
        # print(f'No NAN Percentage: {no_nan_percent_enkf * 100: .2f}%')

        # set model
        model_list = set_models(args)
        if args.v == 'Affine-ydagger':
            model_Avhat, model_Ayhat, model_Aydag, infl_model, local_model, st_model1, st_model2 = model_list
        else:
            model, infl_model, local_model, st_model1, st_model2 = model_list

        # optimizer
        optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)

        # load checkpoint
        if args.cp_load_path != "no":
            load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)

        # test
        print("Test NN Results")
        loss_list_nn = []
        mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn, loc_tensor = \
            test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'testing/test_{args.N}')
        print(f"RMSE: {mean_rmse_nn:.3f} ± {std_rmse_nn:.3f}")
        print(f"RRMSE: {mean_rrmse_nn:.3f} ± {std_rrmse_nn:.3f}")
        print(f"RMV: {mean_rmv_nn:.3f} ± {std_rmv_nn:.3f}")
        print(f"CRPS: {mean_crps_nn:.3f} ± {std_crps_nn:.3f}")
        print(f'No NAN Percentage: {no_nan_percent_nn * 100: .2f}%')
        
        if args.no_localization:
            loc_mean = loc_tensor
            loc_std = loc_tensor
        else:
            loc_mean = torch.mean(loc_tensor, dim=(0,1)) 
            loc_std = torch.std(loc_tensor, dim=(0,1)) 
            
        # save results
        tensor_dict = {
            # 'enkf':{
            #     'mean_rmse':mean_rmse_enkf,
            #     'std_rmse':std_rmse_enkf,
            #     'mean_rmv':mean_rmv_enkf,
            #     'std_rmv':std_rmv_enkf,
            #     'valid_percent':no_nan_percent_enkf,
            # },
            'nn':{
                'mean_rmse':mean_rmse_nn,
                'std_rmse':std_rmse_nn,
                'mean_rrmse':mean_rrmse_nn,
                'std_rrmse':std_rrmse_nn,
                'mean_rmv':mean_rmv_nn,
                'std_rmv':std_rmv_nn,
                'mean_crps':mean_crps_nn,
                'std_crps':std_crps_nn,
                'valid_percent':no_nan_percent_nn,
                'loc_mean':loc_mean,
                'loc_std':loc_std,
                'loc_diff_dist':args.diff_dist,
            },
            'cp_load_path': args.cp_load_path,
            'sigma_y': args.sigma_y,
        }
        
        # print(torch.mean((ens_tensor_enkf.mean(dim=2) - ens_tensor_nn.mean(dim=2))**2, dim=(1,2))[:100])
        
        if args.cp_load_path != "no":
            if args.zero_infl:
                torch.save(tensor_dict, os.path.join(folder_name, f"output_records_zero_infl_{args.N}.pt"))
            else:
                torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))


