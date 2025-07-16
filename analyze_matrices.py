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

# def test_model(loader, model_list, args, infl=1, H_info=None, plot_figures=True, fig_name='example_fig'):
#     if args.v == 'Affine-ydagger':
#         model_Avhat, model_Ayhat, model_Aydag, infl_model, local_model, st_model1, st_model2 = model_list
#         for name, param in model_Avhat.named_parameters():
#             print(name, param.shape)
#             print(param.data)    # shows the actual tensor values
#             print()
#     else:
#         model, infl_model, local_model, st_model1, st_model2 = model_list

#     m = args.N
    
#     if args.dataset == "lorenz63":
#         forward_fun = L63.forward
#     elif args.dataset == "lorenz96":
#         forward_fun = L96.forward
#     elif args.dataset == "ks":
#         forward_fun = etd_rk4_wrapper(device = args.device, dt = args.dt / args.dt_iter)
#     else:
#         raise NotImplementedError

#     if H_info is None:
#         H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
#     else:
#         H_fun, H = H_info
        
#     # loc_mat_vy = dist2coeff(args.Lvy, radius=4).unsqueeze(0)
#     # loc_mat_yy = dist2coeff(args.Lyy, radius=4).unsqueeze(0)

#     results = []

#     with torch.no_grad():
#         for batch_ind, batch_v in enumerate(loader):
#             batch_v = batch_v.to(device=args.device)

#             # Sample from prior
#             ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
#             ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

#             print(ens_v_a[-1, :, :])  # Print the last ensemble member's state

#             # Store everything
#             ens_list = [ens_v_a]
#             loc_records = []

#             # Iterate over timesteps in batch
#             for i in range(len(batch_v) - 1):
#                 t_start = time.time()
#                 # get next observation
#                 obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
#                 obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)

#                 # forecast step
#                 ens_v_a = ens_v_a.reshape(-1, args.ori_dim)
#                 for j in range(args.dt_iter):
#                     if args.dataset == 'ks':
#                         ens_v_a = forward_fun(ens_v_a, None, args.dt / args.dt_iter)
#                     else:
#                         ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
#                                 args.dt / args.dt_iter)
#                 ens_v_f = ens_v_a.view(-1, m, args.ori_dim)
                
#                 # add forward noise
#                 ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

#                 # preparation for individual ensemble data
#                 hv = H_fun(ens_v_f)
                
#                 # ens_v_a, K = EnKF_analysis(ens_v_f, hv, obs_y, args.sigma_y, a_method="PertObs")
#                 B, N, D = ens_v_f.shape
#                 d = hv.shape[2]
                
#                 # generate a random variable for the observation noise
#                 r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
                
#                 ens_i = obs_y - hv - r
                
#                 # for the ensemble dataset and observations
#                 mean_hv = torch.mean(hv, dim=1, keepdim=True).expand(-1, N, -1)
#                 mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True).expand(-1, N, -1)
                
#                 if args.v == 'CorrTerms':
#                     # st and following nn inputs
#                     if args.st_type == 'state_only':
#                         ens_nn_output = st_model1(ens_v_f)
#                         nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
#                                             ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
#                         if args.obs_in_loc:
#                             local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
#                         else:
#                             local_nn_input = ens_nn_output
#                         infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + args.st_output_dim)
#                     elif args.st_type == 'separate':
#                         ens_nn_output = st_model1(ens_v_f)
#                         ens_o_nn_output = st_model2(hv)                
#                         nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
#                                             ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
#                         if args.obs_in_loc:
#                             local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output, ens_o_nn_output], dim=-1)
#                         else:
#                             local_nn_input = torch.cat([ens_nn_output, ens_o_nn_output], dim=-1)
#                         infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + args.st_output_dim)
#                     elif args.st_type == 'joint':
#                         ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
#                         nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
#                                             ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
#                         if args.obs_in_loc:
#                             local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
#                         else:
#                             local_nn_input = ens_nn_output
#                         infl_nn_input = torch.cat([ens_v_f, ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.ori_dim + 2 * args.st_output_dim)
                        
#                     # execute model
#                     nn_output = model(nn_input).view(hv.shape[0], hv.shape[1], -1)
#                     infl_output = infl_model(infl_nn_input).view(B, N, -1)
#                     if args.zero_infl:
#                         Vnn1 = ens_v_f
#                     else:
#                         Vnn1 = ens_v_f + infl_output
                    
#                     # Vnn1 = ens_v_f
#                     Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, :D]
#                     Ynn = hv - mean_hv + nn_output[:, :, D:]
#                     R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                    
#                     # get localization matrices
#                     if args.no_localization:
#                         loc_mat_vy = torch.ones(B, D, d, device=args.device)
#                         loc_mat_yy = torch.ones(B, d, d, device=args.device)
#                     else:
#                         loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * args.loc_max_val
#                         loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
#                         loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
#                         loc_records.append(loc_nn_output)
                    
#                     # Kalman Gain
#                     K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
#                     K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
#                     K = torch.bmm(K1, torch.inverse(K2))
#                     ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
                    
#                     ens_v_a = post_process(ens_v_a, infl=infl)
#                 elif args.v == 'EtE':
#                     if args.st_type == 'state_only':
#                         ens_nn_output = st_model1(ens_v_f)
#                         nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
#                                             ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
#                     elif args.st_type == 'separate':
#                         ens_nn_output = st_model1(ens_v_f)
#                         ens_o_nn_output = st_model2(hv)                
#                         nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
#                                             ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
#                     elif args.st_type == 'joint':
#                         ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
#                         nn_input = torch.cat([ens_v_f, hv, obs_y.expand(-1, N, -1), 
#                                             ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                        
#                     # execute model
#                     nn_output = model(nn_input).view(B, N, -1)
#                     ens_v_a = nn_output
#                 elif args.v == 'LearnK':
#                     r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
#                     obs_plus_noise = hv + r
#                     # s_v = st_model1(ens_v_f)
#                     # s_h = st_model2(obs_plus_noise)
#                     s_v_h = st_model1(torch.cat([ens_v_f, hv], dim = -1))
#                     nn_input = torch.cat([
#                         s_v_h,
#                         obs_y.squeeze(1)
#                     ], dim = -1)
#                     nn_output = model(nn_input)
#                     K = nn_output.view(-1, args.ori_dim, args.obs_dim)
#                     ens_v_a = ens_v_f + torch.bmm(ens_i, K.transpose(1, 2))
#                 elif args.v == 'Affine':
#                     r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
#                     obs_plus_noise = hv + r
#                     s_v_h = st_model1(torch.cat([ens_v_f, hv], dim = -1))
#                     nn_input = torch.cat([
#                         s_v_h,
#                         obs_y.squeeze(1)
#                     ], dim = -1)
#                     nn_output = model(nn_input).view(-1, args.output_dim)
#                     A_mat = nn_output[:, :args.ori_dim**2].view(B, args.ori_dim, args.ori_dim)
#                     B_mat = nn_output[:, args.ori_dim**2: args.ori_dim**2+ args.ori_dim*args.obs_dim].view(B, args.ori_dim, args.obs_dim)
#                     a_vec = nn_output[:, -args.ori_dim:].view(B, args.ori_dim)

#                     Av = torch.bmm(A_mat, ens_v_f.permute(0, 2, 1))
#                     Av = Av.permute(0, 2, 1)
#                     obs_plus_noise = hv + r
#                     By = torch.bmm(B_mat, obs_plus_noise.permute(0, 2, 1))
#                     By = By.permute(0, 2, 1)

#                     a_exp = a_vec.unsqueeze(1)
#                     a_exp = a_exp.expand(-1, N, -1)

#                     ens_v_a = Av + By + a_exp + ens_v_f
#                 elif args.v == 'Affine-ydagger':
#                     r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
#                     obs_plus_noise = hv + r
#                     s_v_h = st_model1(torch.cat([ens_v_f, hv], dim = -1))
#                     nn_input = torch.cat([
#                         s_v_h,
#                         obs_y.squeeze(1)
#                     ], dim = -1)
#                     # nn_output = model(nn_input).view(-1, args.output_dim)
#                     avhat_output = model_Avhat(nn_input).view(B, args.ori_dim, args.ori_dim)
#                     ayhat_output = model_Ayhat(nn_input).view(B, args.ori_dim, args.obs_dim)
#                     aydag_output = model_Aydag(nn_input).view(B, args.ori_dim, args.obs_dim)
#                     # avec_output = model_Avec(nn_input).view(B, args.ori_dim).unsqueeze(1)

#                     Av = torch.bmm(avhat_output, ens_v_f.permute(0, 2, 1))
#                     Av = Av.permute(0, 2, 1)
#                     obs_plus_noise = hv + r
#                     Ayhat = torch.bmm(ayhat_output, obs_plus_noise.permute(0, 2, 1))
#                     Ayhat = Ayhat.permute(0, 2, 1)
#                     # multiply aydag with the true observation
#                     Aydag_y = torch.bmm(aydag_output, obs_y.permute(0, 2, 1))
#                     Aydag_y = Aydag_y.permute(0, 2, 1)
#                     ens_v_a = Av + Ayhat + Aydag_y + ens_v_f

#                 ens_v_a = torch.clamp(ens_v_a, min=-args.clamp, max=args.clamp)

#                 ens_list.append(ens_v_a)

#                 Vnn1 = ens_v_f
#                 Vnn2 = ens_v_f - mean_ens_v_f
#                 Ynn = hv - mean_hv
#                 R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                
#                 # get localization matrices
#                 if args.no_localization:
#                     K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
#                     K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R * (N - 1)
#                 else:
#                     loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * args.loc_max_val
#                     loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
#                     loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
                    
#                     K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
#                     K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
                
#                 # Kalman Gain
#                 K = torch.bmm(K1, torch.inverse(K2))


#                 diff_yhat = torch.norm(K - ayhat_output, p = 'fro', dim=(1, 2))  # (B,)
#                 diff_ydag = torch.norm(K - aydag_output, p = 'fro', dim=(1, 2))  # (B,)
#                 B, d, _ = avhat_output.shape
#                 I = torch.eye(d, device=avhat_output.device, dtype=avhat_output.dtype)  # (d,d)
#                 I = I.unsqueeze(0).expand(B, d, d)                                       # (B,d,d)
#                 diff_avhat = torch.norm(I - avhat_output, p='fro', dim =(1, 2))  # (B,)

#                 results.append(torch.stack([diff_yhat, diff_ydag, diff_avhat], dim=1))  # (B, 3)
                
#             # Concat outputs
#             ens_tensor = torch.stack(ens_list)
#             if args.v == "EtE" or args.v == 'LearnK' or args.v == 'Affine' or args.v == 'Affine-ydagger':
#                 loc_tensor = None
#             else:
#                 if args.no_localization:
#                     loc_tensor = torch.empty(1)
#                 else:
#                     loc_tensor = torch.stack(loc_records)
            
#             # Loss functions
#             # absolute rmse
#             crps_tensor = torch.mean(compute_es(ens_states=ens_tensor, true_states=batch_v, norm_p=1), dim=0) / torch.mean(torch.norm(batch_v, p=2, dim=2), dim=0)
#             rmse_tensor = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)), dim=0)
#             rms_tensor = torch.mean(torch.sqrt(torch.mean((batch_v) ** 2, dim=2)), dim=0)
#             rrmse_tensor = rmse_tensor / rms_tensor
#             # relative rmse
#             # rmse_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)) / torch.sqrt(torch.mean((batch_v) ** 2, dim=2))
#             rmv_tensor = torch.mean(torch.sqrt(N / (N-1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3))),dim=0)
            
#             if batch_ind == 0:
#                 rmse_tensor_all, rmv_tensor_all, rrmse_tensor_all, crps_tensor_all = rmse_tensor, rmv_tensor, rrmse_tensor, crps_tensor
#             else:
#                 # Concatenate tensors along the first dimension
#                 rmse_tensor_all = torch.cat((rmse_tensor_all, rmse_tensor))
#                 rmv_tensor_all = torch.cat((rmv_tensor_all, rmv_tensor))
#                 rrmse_tensor_all = torch.cat((rrmse_tensor_all, rrmse_tensor))
#                 crps_tensor_all = torch.cat((crps_tensor_all, crps_tensor))
#         result = torch.stack(results, dim=0)
#         print(result.shape)
#         print(ens_tensor.shape)

#         plot_particle_trajectories_with_histograms(particles=ens_tensor[:,1,:,:], 
#                                                     true_traj=batch_v[:,0,:], 
#                                                     # observation=observations[:,-2,:],
#                                                     observation=None,
#                                                     dim_indices=[0, 1, 2, 3],
#                                                     start_time=args.test_steps-100,
#                                                     end_time=args.test_steps, 
#                                                     mode='quantile',
#                                                     save_fig=True,
#                                                     save_pdf=True,
#                                                     save_name=fig_name,
#                                                     hist_step=1,
#                                                     fontsize=None)
        
#         # non-nan trajs
#         nan_mask = torch.isnan(rrmse_tensor_all) 
#         valid_B_mask = ~nan_mask
        
#         mean_rrmse, std_rrmse = get_mean_std(rrmse_tensor_all[valid_B_mask])
#         mean_rmse, std_rmse = get_mean_std(rmse_tensor_all[valid_B_mask])
#         mean_rmv, std_rmv = get_mean_std(rmv_tensor_all[valid_B_mask])
#         mean_crps, std_crps = get_mean_std(crps_tensor_all[valid_B_mask])
        
#         no_nan_percent = torch.sum(valid_B_mask) / args.test_traj_num

#     return mean_rmse, std_rmse, mean_rmv, std_rmv, mean_rrmse, std_rrmse, mean_crps, std_crps, no_nan_percent, loc_tensor

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

                