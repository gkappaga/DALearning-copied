#!/bin/bash

cd ..

# # Lorenz 63
# python finetune.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset lorenz63 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 1 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --cp_load_path save/2025-04-10_17-18lorenz63_1.0_10_60_8192_norm_EnST_joint/cp_1000.pth \
#     --no_localization 

# python finetune.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset lorenz63 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --cp_load_path save/2025-04-10_19-40lorenz63_0.7_10_60_8192_norm_EnST_joint/cp_1000.pth \
#     --no_localization 

# # Lorenz 96
#'save/2026-04-29_22-16lorenz96_None_10_60_8192_nl2_joint_Affine-ydagger_3k_sigma^2_randomH_l96_sigma_y_1_2_infl_st250k/cp_3000.pth'
python finetune.py \
    --epochs 100 \
    --save_epoch 50 \
    --dataset lorenz96 \
    --v Affine-ydagger \
    --random_h True \
    --random_noise \
    --train_steps 60 \
    --train_traj_num 8192 \
    --seed 42 \
    --learning_rate 1e-4 \
    --sigma_y 1 \
    --loss_type 'es' \
    --num_loader_workers 16 \
    --cp_load_path 'important_save/2026-06-03_19-46lorenz96_None_10_60_8192_es_joint_Affine-ydagger_1k_randomH_l96_sigma_y_1_2_infl_st250k/cp_1000.pth'


# python finetune.py \
#     --epochs 35 \
#     --save_epoch 35 \
#     --dataset lorenz63 \
#     --v Affine-ydagger \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --random_noise \
#     --random_h True \
#     --cp_load_path 'save/2026-03-29_22-38lorenz63_None_10_60_8192_nl2_joint_Affine-ydagger_3k_sigma^2_randomH_l63_sigma_y_1_2/cp_3000.pth'

# python finetune.py \
#     --epochs 35 \
#     --save_epoch 35 \
#     --dataset ks \
#     --v Affine-ydagger \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --random_noise \
#     --random_h True \
#     --cp_load_path 'save/2026-04-03_16-51ks_None_10_60_8192_nl2_joint_Affine-ydagger_3k_sigma^2_randomH_ks_sigma_y_1_2/cp_3000.pth'


# python finetune.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset lorenz96 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --cp_load_path save/2025-04-10_13-18lorenz96_0.7_10_60_8192_norm_EnST_joint/cp_1000.pth

# # # KS
# python finetune.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset ks \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 1 \
#     --seed 42 \
#     --learning_rate 5e-5 \
#     --cp_load_path save/2025-04-09_18-18ks_1.0_10_60_8192_norm_EnST_joint/cp_1000.pth

# python finetune.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset ks \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --learning_rate 5e-5 \
#     --cp_load_path save/2025-04-10_01-52ks_0.7_10_60_8192_norm_EnST_joint/cp_1000.pth



