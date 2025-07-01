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

python train.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --v 'Affine' \
    --epochs 100 \
    --save_epoch 10 \
    --mc_penalty True \
    --lambda1 0 \
    --lambda2 0.00005

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 

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


