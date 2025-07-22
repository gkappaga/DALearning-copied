import datetime
import os

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint
from utils import partial_obs_operator, get_dataloader, redirect_output

from training_utils import train_model, test_model, set_models


if __name__ == "__main__":
    args = get_parameters()

    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    
    # redirect output
    with redirect_output(args.save_folder, filename="output.txt"):
        # save_folder
        folder_name = args.save_folder
        
        # torch.cuda.set_device(args.device)
        for key, value in vars(args).items():
            print(f"{key}: {value}")

        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        train_loader, test_loader = get_dataloader(args)

        # set models
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
            # for param in st_model1.parameters():
            #     param.requires_grad = False
            # for param in st_model2.parameters():
            #     param.requires_grad = False

        # training
        train_loss_list = []
        test_rmse_list = []
        test_rrmse_list = []
        test_epochs = []
        train_mean_pen_list = []
        train_cov_pen_list = []
        train_orig_pen_list = []
        if args.test_only:
            print("Test Only")
            rmse_list, rrmse_list = [], []
            for i in range(args.test_rounds):
                mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn, loc_tensor = \
                        test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/test_only_{args.N}')
                rmse_list.append(mean_rmse_nn)
                rrmse_list.append(mean_rrmse_nn)
            print("Average RMSE:", torch.mean(torch.tensor(rmse_list)))
            print("Average R-RMSE:", torch.mean(torch.tensor(rrmse_list)))
        else:
            print("Training Start")
            mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn, loc_tensor = \
                        test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/test_only_{args.N}_0')
            print(f"RMSE: {mean_rmse_nn:.3f} ± {std_rmse_nn:.3f}")
            print(f"RRMSE: {mean_rrmse_nn:.3f} ± {std_rrmse_nn:.3f}")
            print(f"RMV: {mean_rmv_nn:.3f} ± {std_rmv_nn:.3f}")
            print(f"CRPS: {mean_crps_nn:.3f} ± {std_crps_nn:.3f}")
            print(f'No NAN Percentage: {no_nan_percent_nn * 100: .2f}%')
            test_epochs.append(0)
            test_rmse_list.append(mean_rmse_nn)
            test_rrmse_list.append(mean_rrmse_nn)
            for epoch in range(1, 1 + args.epochs):
                if args.mc_penalty:
                    train_loss, mean_pen, cov_pen, orig_pen = train_model(epoch, train_loader, model_list, optimizer, scheduler, args, H_info=H_info)
                    train_mean_pen_list.append(mean_pen)
                    train_cov_pen_list.append(cov_pen)
                    train_orig_pen_list.append(orig_pen)
                else:
                    train_loss = train_model(epoch, train_loader, model_list, optimizer, scheduler, args, H_info=H_info)
                train_loss_list.append(train_loss)
                
                if epoch % args.save_epoch == 0:
                    mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, mean_rrmse_nn, std_rrmse_nn, mean_crps_nn, std_crps_nn, no_nan_percent_nn, loc_tensor = \
                        test_model(test_loader, model_list, args, H_info=H_info, plot_figures=True, fig_name=f'{folder_name}/test_only_{args.N}_{epoch}')
                    print(f"RMSE: {mean_rmse_nn:.3f} ± {std_rmse_nn:.3f}")
                    print(f"RRMSE: {mean_rrmse_nn:.3f} ± {std_rrmse_nn:.3f}")
                    print(f"RMV: {mean_rmv_nn:.3f} ± {std_rmv_nn:.3f}")
                    print(f"CRPS: {mean_crps_nn:.3f} ± {std_crps_nn:.3f}")
                    print(f'No NAN Percentage: {no_nan_percent_nn * 100: .2f}%')
                    test_epochs.append(epoch)
                    test_rmse_list.append(mean_rmse_nn)
                    test_rrmse_list.append(mean_rrmse_nn)
                    if args.mc_penalty:
                        train_records = {"train_loss": train_loss_list, "test_loss": test_rmse_list, "test_rrmse": test_rrmse_list, "test_epochs": test_epochs, "train_mean_pen": train_mean_pen_list, "train_cov_pen": train_cov_pen_list, "train_orig_pen": train_orig_pen_list}
                    else:
                        train_records = {"train_loss": train_loss_list, "test_loss": test_rmse_list, "test_rrmse": test_rrmse_list, "test_epochs": test_epochs}
                    torch.save(train_records, os.path.join(folder_name, f"training_records.pt"))
                    save_checkpoint(model_list, optimizer, scheduler, filename=os.path.join(folder_name, f"cp_{epoch}.pth"))




