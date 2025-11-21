#!/bin/bash

cd ..

# lorenz 63
dataset="lorenz63"

sigma_y=1
seed=42
save_dir="2025-08-08_20-06lorenz63_None_10_60_8192_nl2_joint_Affine-ydagger_3k_sigma^2_l63"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --v 'Affine-ydagger' \
        --seed $seed \
        --no_localization \
        --cp_load_path save/${save_dir}/cp_3000.pth
done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-10_19-40lorenz63_0.7_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --no_localization \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# # # lorenz 96
# dataset="lorenz96"

# sigma_y=1
# seed=42
# save_dir="2025-04-10_09-31lorenz96_1.0_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-10_13-18lorenz96_0.7_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# # ks
# dataset="ks"

# sigma_y=1
# seed=42
# save_dir="2025-04-09_18-18ks_1.0_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-10_01-52ks_0.7_10_60_8192_norm_EnST_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/cp_1000.pth
# done

