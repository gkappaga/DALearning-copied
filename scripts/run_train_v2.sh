#!/bin/bash

cd ..

python train.py \
    --dataset lorenz63 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --v 'Affine-ydagger' \
    --epochs 1000 \
    --save_epoch 50 \
    --loss_type 'nl2' \
    --lr_decay_epochs 200,300,400,500,600,700,800,900 \
    --lr_decay_rate 0.7 \
# python train.py \
#     --dataset lorenz96 \
#     --learning_rate 1e-4 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --loss_type es \
#     --lr_decay_epochs 400,800,1200,1600,2000,2400 \
#     --lr_decay_rate 0.7 \
#     --epochs 3000 \
#     --v EtE \
#     --warm_up

# python train.py \
#     --dataset lorenz96 \
#     --learning_rate 1e-4 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --loss_type nes \
#     --lr_decay_epochs 400,800,1200,1600,2000,2400 \
#     --lr_decay_rate 0.7 \
#     --epochs 3000 \
#     --v EtE \
#     --warm_up

# python train.py \
#     --dataset lorenz96 \
#     --learning_rate 1e-2 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --loss_type tnes \
#     --lr_decay_epochs 400,800,1200,1600,2000,2400 \
#     --lr_decay_rate 0.7 \
#     --epochs 3000 \
#     --v EtE \
#     --warm_up