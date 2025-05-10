#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fn_dict
from torchvision.utils import save_image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from scene.direct_light_map import DirectLightMap
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision
from utils.graphics_utils import rgb_to_srgb, srgb_to_rgb
import json

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, pbr_kwargs=None, static_dataset=None, test_light_folder = None):

    if not gaussians.use_pbr:
        raise NotImplemented
    
    render_path = os.path.join(model_path, "scale_albedo_static_preview", "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)

    static_albedo_folder = os.path.join(static_dataset, "albedo")
    static_relight_folder = os.path.join(static_dataset, test_light_folder)

    albedo_gt_list =[]
    albedo_list = []

    render_fn = render_fn_dict[args.type]
    for render_idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        results = render_fn(view, gaussians, pipeline, background, dict_params=pbr_kwargs)
        rendering = torch.clamp(results["base_color"], 0.0, 1.0)
        torchvision.utils.save_image(rendering.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_train_srgb{render_idx}_{view.image_name}.png'))
        
        rendering_linear = srgb_to_rgb(rendering)
        torchvision.utils.save_image(rendering_linear.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_train_linear{render_idx}_{view.image_name}.png'))

        target_size = rendering.shape[1:]  # (H, W)

        view_path = view.image_name
        render_name = os.path.basename(view_path).split(".")[0]
        albedo_gt_path = [os.path.join(static_albedo_folder, f) for f in os.listdir(static_albedo_folder) if render_name in f][0]
        albedo_gt = read_image(albedo_gt_path).float()[:3]
        # Resize albedo_gt to match rendering resolution
        albedo_gt = F.interpolate(albedo_gt.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        albedo_gt /= 255.0  # normalize to [0,1]

        torchvision.utils.save_image(albedo_gt.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_gt_static_srgb{render_idx}_{view.image_name}.png'))
        albedo_gt = srgb_to_rgb(albedo_gt)
        torchvision.utils.save_image(albedo_gt.clamp(0.0, 1.0), os.path.join(render_path, f'albedo_gt_static_linear{render_idx}_{view.image_name}.png'))

        #read mask from renders
        relight_gt_path = [os.path.join(static_relight_folder, f) for f in os.listdir(static_relight_folder) if render_name in f][0]
        mask = read_image(relight_gt_path).float()[3:4]
        # Resize mask to match rendering resolution
        mask = F.interpolate(mask.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        mask /= 255.0  # normalize to [0,1]
        # torchvision.utils.save_image(mask.clamp(0.0, 1.0), os.path.join(render_path, f'mask{render_idx}.png'))
    
        print("RENDERING ALBEDO SHAPES, ", albedo_gt.shape, rendering_linear.shape)
        
        albedo_gt_list.append(albedo_gt.permute(1, 2, 0)[mask[0] > 0])
        albedo_list.append(rendering_linear.permute(1, 2, 0)[mask[0] > 0])
            
        albedo_gts = torch.cat(albedo_gt_list, dim=0).cuda()
        albedo_ours = torch.cat(albedo_list, dim=0)
        albedo_scale_json = {}
        albedo_scale_json["0"] = [1.0, 1.0, 1.0]
        albedo_scale_json["1"] = [(albedo_gts/albedo_ours.clamp_min(1e-6))[..., 0].median().item()] * 3
        albedo_scale_json["2"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).median(dim=0).values.tolist()
        albedo_scale_json["3"] = (albedo_gts/albedo_ours.clamp_min(1e-6)).mean(dim=0).tolist()
        print("Albedo scales:\n", albedo_scale_json)
            
        with open(os.path.join(args.model_path, "albedo_scale_linear_static.json"), "w") as f:
            json.dump(albedo_scale_json, f)


        

def render_sets(dataset : ModelParams, pipeline : PipelineParams, skip_train : bool, skip_test : bool, static_dataset=None, test_light_folder = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
        scene = Scene(dataset, gaussians, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if args.checkpoint:
            print("Create Gaussians from checkpoint {}".format(args.checkpoint))
            iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)
        elif scene.loaded_iter:
            gaussians.load_ply(os.path.join(dataset.model_path,
                                            "point_cloud",
                                            "iteration_" + str(scene.loaded_iter),
                                            "point_cloud.ply"))
            iteration = scene.loaded_iter
        else:
            gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
            iteration = scene.loaded_iter

        pbr_kwargs = dict()
        if iteration is not None and gaussians.use_pbr:
            gaussians.update_visibility(args.sample_num)
        
            pbr_kwargs['sample_num'] = args.sample_num
            print("Using global incident light for regularization.")
            direct_env_light = DirectLightMap(args.env_resolution)
            
            if args.checkpoint:
                env_checkpoint = os.path.dirname(args.checkpoint) + "/env_light_" + os.path.basename(args.checkpoint)
                print("Trying to load global incident light from ", env_checkpoint)
                if os.path.exists(env_checkpoint):
                    direct_env_light.create_from_ckpt(env_checkpoint, restore_optimizer=True)
                    print("Successfully loaded!")
                else:
                    print("Failed to load!")
                pbr_kwargs["env_light"] = direct_env_light

        
        # if not skip_train:
        #      render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, pbr_kwargs)
        print(len(scene.getTestCameras()), "VIEWS LEN!!!")
        if not skip_test:
             render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, pbr_kwargs,static_dataset, test_light_folder)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf'], default='render')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--static_source_path", default="", type=str)
    parser.add_argument("--test_light_folder", default="", type=str)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args.skip_train, args.skip_test, 
                args.static_source_path, args.test_light_folder)