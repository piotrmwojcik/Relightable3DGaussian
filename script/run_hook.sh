#!/bin/bash

root_dir="datasets/nerf_synthetic/"
object="hook150_v3_transl_statictimestep1"
list="chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330 small_harbour_sunset_4k_32x16_rot270"

for i in $list; do
    python train.py --eval \
        -s datasets/nerf_synthetic/$object/ \
        -m output/our_ds/$i/3dgs \
        --lambda_normal_render_depth 0.01 \
        --lambda_normal_smooth 0.01 \
        -envmap_prefix $i \
        --lambda_mask_entropy 0.1 \
        --save_training_vis \
        --lambda_depth_var 1e-2 \
        --resolution 2

    python eval_nvs.py --eval \
        -m output/our_ds/${i}/3dgs \
        -c output/our_ds/${i}/3dgs/chkpnt30000.pth \
        --resolution 2

    python train.py --eval \
        -s datasets/nerf_synthetic/$object/ \
        -envmap_prefix $i \
        -m output/our_ds/$i/neilf \
        -c output/our_ds/$i/3dgs/chkpnt30000.pth \
        --save_training_vis \
        --position_lr_init 0.000016 \
        --position_lr_final 0.00000016 \
        --normal_lr 0.001 \
        --sh_lr 0.00025 \
        --opacity_lr 0.005 \
        --scaling_lr 0.0005 \
        --rotation_lr 0.0001 \
        --iterations 40000 \
        --lambda_base_color_smooth 0 \
        --lambda_roughness_smooth 0 \
        --lambda_light_smooth 0 \
        --lambda_light 0.01 \
        -t neilf --sample_num 64 \
        --save_training_vis_iteration 200 \
        --lambda_env_smooth 0.01 \
        --resolution 2
    
    python eval_nvs.py --eval \
        -m output/our_ds/${i}/neilf \
        -c output/our_ds/${i}/neilf/chkpnt40000.pth \
        -t neilf \
        --resolution 2
done
