#!/bin/bash

cd ..

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v 'LearnK' \
#     --epochs 500 \
#     --save_epoch 50

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --v 'LearnK' \
#     --epochs 1000 \
#     --save_epoch 50 \
#     --loss_type 'nl2' \

python train.py \
    --dataset lorenz96 \
    --N 10 \
    --seed 42 \
    --v 'Affine-ydagger' \
    --epochs 3000 \
    --save_epoch 100 \
    --loss_type 'nl2' \
    --lr_decay_epochs 400,800,1200,1600,2000,2400,2800 \
    --lr_decay_rate 0.7 \
    --random_noise \
    --suffix '_3k_sigma^2_randomH_l96' \
    --random_h True
# python finetune.py \
#     --dataset lorenz96 \
#     --seed 42 \
#     --v 'Affine-ydagger' \
#     --epochs 500 \
#     --save_epoch 50 \
#     --loss_type 'nl2' \
#     --cp_load_path save/2025-07-21_22-00lorenz96_1.0_10_60_8192_nl2_joint_Affine-ydagger/cp_1000.pth \

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 1 \
#     --no_localization 

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 1 \
#     --no_localization 

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 16 \
#     --adjust_lr

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 20 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 1 \
#     --adjust_lr

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 20 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 8 \
#     --adjust_lr

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 20 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 16 \
#     --adjust_lr


# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --adjust_lr


