#!/bin/bash


#!/bin/bash

CUDA_VISIBLE_DEVICES=1
RESOLUTION=2



############### DIFFUSE!!!

SOURCE_PATHS=(
#  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight-spec32/hook150_v5_spec32_statictimestep1"
#  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight-spec32/mouse150_v5_spec32_statictimestep1"
 "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight-spec32/jumpingjacks150_v5_spec32_statictimestep75"
#  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1"
#  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight-spec32/standup150_v5_spec32_statictimestep75"
)

LIGHT_COMBINATIONS=(
    "dam_wall_4k_32x16_rot90 small_harbour_sunset_4k_32x16_rot270 damwall harbour"
    # "golden_bay_4k_32x16_rot330 dam_wall_4k_32x16_rot90 goldenbay damwall"
    # "chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330 chapelday goldenbay"
)



for LIGHT_ENTRY in "${LIGHT_COMBINATIONS[@]}"; do
    read TRAIN_LIGHT TEST_LIGHT TRAIN_NAME TEST_NAME <<< "$LIGHT_ENTRY"
    
    for SOURCE_PATH in "${SOURCE_PATHS[@]}"; do
        SCENE_NAME=$(basename "$SOURCE_PATH")

        OUTPUT_PATH="output_specular32/${TRAIN_NAME}_${TEST_NAME}/${SCENE_NAME}_r${RESOLUTION}"
        CKPT_PATH="output_specular32/${TRAIN_NAME}_${TEST_NAME}/${SCENE_NAME}_r${RESOLUTION}"


        echo "Processing $SOURCE_PATH -> $OUTPUT_PATH"
        echo "Train: $TRAIN_LIGHT ($TRAIN_NAME) | Test: $TEST_LIGHT ($TEST_NAME)"


        # python train.py --eval \
        #     -s $SOURCE_PATH \
        #     -m $OUTPUT_PATH/3dgs \
        #     --lambda_normal_render_depth 0.01 \
        #     --lambda_normal_smooth 0.01 \
        #     -envmap_prefix $TRAIN_LIGHT \
        #     --lambda_mask_entropy 0.1 \
        #     --save_training_vis \
        #     --lambda_depth_var 1e-2 \
        #     --resolution=$RESOLUTION


        # python train.py --eval \
        #     -s $SOURCE_PATH \
        #     -envmap_prefix $TRAIN_LIGHT \
        #     -m $OUTPUT_PATH/neilf \
        #     -c $CKPT_PATH/3dgs/chkpnt30000.pth \
        #     --save_training_vis \
        #     --position_lr_init 0.000016 \
        #     --position_lr_final 0.00000016 \
        #     --normal_lr 0.001 \
        #     --sh_lr 0.00025 \
        #     --opacity_lr 0.005 \
        #     --scaling_lr 0.0005 \
        #     --rotation_lr 0.0001 \
        #     --iterations 40000 \
        #     --lambda_base_color_smooth 0 \
        #     --lambda_roughness_smooth 0 \
        #     --lambda_light_smooth 0 \
        #     --lambda_light 0.01 \
        #     -t neilf --sample_num 64 \
        #     --save_training_vis_iteration 200 \
        #     --lambda_env_smooth 0.01 \
        #     --resolution=$RESOLUTION
        
        # python eval_nvs.py --eval \
        #     -m $OUTPUT_PATH/neilf/ \
        #     -c $OUTPUT_PATH/neilf/chkpnt40000.pth \
        #     -t neilf \
        #     --resolution=$RESOLUTION

        # python scale_albedo.py --eval \
        #     -m $OUTPUT_PATH/neilf/ \
        #     -c $OUTPUT_PATH/neilf/chkpnt40000.pth \
        #     -t neilf \
        #     --resolution=$RESOLUTION \
        #     --static_source_path $SOURCE_PATH \
        #     --test_light_folder $TEST_LIGHT

        python eval_relighting_ours.py --eval \
            -m $OUTPUT_PATH/neilf/ \
            -c $OUTPUT_PATH/neilf/chkpnt40000.pth \
            --resolution=$RESOLUTION \
            --static_source_path $SOURCE_PATH \
            --test_light_folder chapel_day_4k_32x16_rot0 \
            --sample_num 1024
        done
    done


# ############### DIFFUSE!!!

# SOURCE_PATHS=(
# # "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight/jumpingjacks150_v3_tex_statictimestep75"
# #  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight/spheres_cube_dataset_v5_statictimestep1"
# #  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight/standup150_v3_statictimestep75"
# #  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight/hook150_v3_transl_statictimestep1"
# #  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight/mouse150_v2_transl_statictimestep1"
#  "/home/jk/Dynamic-2DGS-relightable/data/d-nerf-relight/spheres_cube_dataset_v8_spec32_statictimestep1"
# )

# LIGHT_COMBINATIONS=(
#     "dam_wall_4k_32x16_rot90 small_harbour_sunset_4k_32x16_rot270 damwall harbour"
#     "golden_bay_4k_32x16_rot330 dam_wall_4k_32x16_rot90 goldenbay damwall"
#     "chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330 chapelday goldenbay"
# )



# for LIGHT_ENTRY in "${LIGHT_COMBINATIONS[@]}"; do
#     read TRAIN_LIGHT TEST_LIGHT TRAIN_NAME TEST_NAME <<< "$LIGHT_ENTRY"
    
#     for SOURCE_PATH in "${SOURCE_PATHS[@]}"; do
#         SCENE_NAME=$(basename "$SOURCE_PATH")

#         OUTPUT_PATH="output_diffuse_noprior/${TRAIN_NAME}_${TEST_NAME}/${SCENE_NAME}_r${RESOLUTION}"
#         CKPT_PATH="output_diffuse/${TRAIN_NAME}_${TEST_NAME}/${SCENE_NAME}_r${RESOLUTION}"


#         echo "Processing $SOURCE_PATH -> $OUTPUT_PATH"
#         echo "Train: $TRAIN_LIGHT ($TRAIN_NAME) | Test: $TEST_LIGHT ($TEST_NAME)"


#         # python train.py --eval \
#         #     -s $SOURCE_PATH \
#         #     -m $OUTPUT_PATH/3dgs \
#         #     --lambda_normal_render_depth 0.01 \
#         #     --lambda_normal_smooth 0.01 \
#         #     -envmap_prefix $TRAIN_LIGHT \
#         #     --lambda_mask_entropy 0.1 \
#         #     --save_training_vis \
#         #     --lambda_depth_var 1e-2 \
#         #     --resolution=$RESOLUTION


#         python train.py --eval \
#             -s $SOURCE_PATH \
#             -envmap_prefix $TRAIN_LIGHT \
#             -m $OUTPUT_PATH/neilf \
#             -c $CKPT_PATH/3dgs/chkpnt30000.pth \
#             --save_training_vis \
#             --position_lr_init 0.000016 \
#             --position_lr_final 0.00000016 \
#             --normal_lr 0.001 \
#             --sh_lr 0.00025 \
#             --opacity_lr 0.005 \
#             --scaling_lr 0.0005 \
#             --rotation_lr 0.0001 \
#             --iterations 40000 \
#             --lambda_base_color_smooth 0 \
#             --lambda_roughness_smooth 0 \
#             --lambda_light_smooth 0 \
#             --lambda_light 0.0 \
#             -t neilf --sample_num 64 \
#             --save_training_vis_iteration 200 \
#             --lambda_env_smooth 0.01 \
#             --resolution=$RESOLUTION
        
        # python eval_nvs.py --eval \
        #     -m $OUTPUT_PATH/neilf/ \
        #     -c $OUTPUT_PATH/neilf/chkpnt40000.pth \
        #     -t neilf \
        #     --resolution=$RESOLUTION

#         python scale_albedo.py --eval \
#             -m $OUTPUT_PATH/neilf/ \
#             -c $OUTPUT_PATH/neilf/chkpnt40000.pth \
#             -t neilf \
#             --resolution=$RESOLUTION \
#             --static_source_path $SOURCE_PATH \
#             --test_light_folder $TEST_LIGHT

#         python eval_relighting_ours.py --eval \
#             -m $OUTPUT_PATH/neilf/ \
#             -c $OUTPUT_PATH/neilf/chkpnt40000.pth \
#             --resolution=$RESOLUTION \
#             --static_source_path $SOURCE_PATH \
#             --test_light_folder $TEST_LIGHT \
#             --sample_num 512

#         done
#     done