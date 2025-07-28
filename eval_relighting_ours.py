import json
import os
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from scene import GaussianModel, Scene
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.envmap import EnvLight
from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr
from scene.utils import load_img_rgb
import warnings
import json
from utils.graphics_utils import srgb_to_rgb, rgb_to_srgb
warnings.filterwarnings("ignore")
from torchvision.io import read_image
import torch.nn.functional as F


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument('-e', '--envmap_path', default=None, help="Env map path")
    parser.add_argument('-bg', "--background_color", type=float, default=1,
                        help="If set, use it as background color")
    parser.add_argument("--static_source_path", default="", type=str)
    parser.add_argument("--test_light_folder", default="", type=str)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load gaussians
    gaussians = GaussianModel(model.sh_degree, render_type="neilf")

    scene = Scene(dataset, gaussians, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)
    else:
        raise NotImplementedError
        
    # deal with each item
    views = scene.getTestCameras()

    static_albedo_folder = os.path.join(args.static_source_path, "albedo")
    static_relight_folder = os.path.join(args.static_source_path, args.test_light_folder)

    task_dict = {
        "env": {
            "capture_list": ["pbr", "pbr_env", "base_color", "global_lights", "normal", "normal_view", "diffuse"],
            "envmap_path": os.path.join(args.static_source_path, f"{args.test_light_folder}.hdr" ),
        },
    }

    print(task_dict["env"]["capture_list"])

    print("EVAL FOR ENVMAP", task_dict["env"]["envmap_path"])

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict['neilf']
    gaussians.update_visibility(args.sample_num)
    
    results_dir = os.path.join(args.model_path, "test_rli")
    task_names = ['env']
    for task_name in task_names:
        task_dir = os.path.join(results_dir, task_name)
        print(task_dir, "TASKKK DIRRRR")
        os.makedirs(task_dir, exist_ok=True)
        light = EnvLight(path=task_dict[task_name]["envmap_path"], scale=1)

        
        
        # if "/air_baloons/" in args.model_path:
        #     gaussians.base_color_scale = torch.tensor([1.3746, 0.6428, 0.7279], dtype=torch.float32, device="cuda")
        # elif "/chair/" in args.model_path:
        #     gaussians.base_color_scale = torch.tensor([1.8865, 1.9675, 1.7410], dtype=torch.float32, device="cuda")
        # elif "/hotdog/" in args.model_path:
        #     gaussians.base_color_scale = torch.tensor([2.6734, 2.0917, 1.2587], dtype=torch.float32, device="cuda")
        # elif "/jugs/" in args.model_path:
        #     # gaussians.base_color_scale = torch.tensor([1.1916, 0.9296, 0.5684], dtype=torch.float32, device="cuda")
        #     gaussians.base_color_scale = torch.tensor([1.0044, 0.9253, 0.7648], dtype=torch.float32, device="cuda")
        # else:
        #     raise NotImplementedError
        with open(os.path.join(args.model_path, "albedo_scale_linear_static.json"), "r") as f:
            albedo_scale_dict = json.load(f)
        base_color_scale = torch.tensor(albedo_scale_dict["2"], dtype=torch.float32, device="cuda")
        gaussians.base_color_scale = base_color_scale #we compute scale in linear, and its used as linear

        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "is_training": False,
            "dict_params": {
                "env_light": light,
                "sample_num": args.sample_num,
            },
        }
        
        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        
        psnr_albedo = 0.0
        ssim_albedo = 0.0
        lpips_albedo = 0.0
        
        
        capture_list = task_dict[task_name]["capture_list"]
        for capture_type in capture_list:
            capture_type_dir = os.path.join(task_dir, capture_type)
            print(capture_type_dir, "Capture dirrrrr")

            os.makedirs(capture_type_dir, exist_ok=True)

        os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "gt_albedo"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "gt_pbr_env"), exist_ok=True)
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]
            
        for idx, view in enumerate(tqdm(views, leave=False)):

            with torch.no_grad():
                render_pkg = render_fn(viewpoint_camera=view, **render_kwargs)


            view_path = view.image_name
            render_name = os.path.basename(view_path).split(".")[0]

            target_size = render_pkg["render"].shape[1:]  # (H, W)

            #read mask from renders
            relight_gt_path = [os.path.join(static_relight_folder, f) for f in os.listdir(static_relight_folder) if render_name in f][0]
            mask = read_image(relight_gt_path).float()[3:4]
            # Resize mask to match rendering resolution
            mask = F.interpolate(mask.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            mask /= 255.0  # normalize to [0,1]
            # torchvision.utils.save_image(mask.clamp(0.0, 1.0), os.path.join(render_path, f'mask{render_idx}.png'))

            #read mask from renders
            gt_image = read_image(relight_gt_path).float()[:3]
            # Resize mask to match rendering resolution
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            gt_image /= 255.0  # normalize to [0,1]
            # torchvision.utils.save_image(mask.clamp(0.0, 1.0), os.path.join(render_path, f'mask{render_idx}.png'))
            gt_image = gt_image * mask + bg * (1 - mask)
            save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))

            # read albedo
            albedo_gt_path = [os.path.join(static_albedo_folder, f) for f in os.listdir(static_albedo_folder) if render_name in f][0]
            gt_albedo = read_image(albedo_gt_path).float()[:3]
            # Resize albedo_gt to match rendering resolution
            gt_albedo = F.interpolate(gt_albedo.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0).cuda()
            gt_albedo /= 255.0  # normalize to [0,1]
            gt_albedo = gt_albedo * mask + bg * (1 - mask)
            save_image(gt_albedo, os.path.join(task_dir, "gt_albedo", f"{idx}.png"))

            
            for capture_type in capture_list:
                if capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type == "normal_view":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["roughness", "diffuse", "specular", "lights", "local_lights", "global_lights", "visibility"]:
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["base_color"]:
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["pbr"]:
                    render_pkg[capture_type] = render_pkg["pbr"] * mask + (1 - mask) * bg
                elif capture_type in ["pbr_env"]:
                    render_pkg[capture_type] = render_pkg["pbr"] * mask + (1 - mask) * render_pkg["env_only"]
                save_image(render_pkg[capture_type], os.path.join(task_dir, capture_type, f"{idx}.png"))
                
            
            gt_image_env = gt_image * mask + render_pkg["env_only"] * (1 - mask)
            save_image(gt_image_env, os.path.join(task_dir, "gt_pbr_env", f"{idx}.png"))
            
            with torch.no_grad():
                psnr_pbr += psnr(torch.nan_to_num(render_pkg['pbr'].clamp(0,1), nan=0.0), gt_image).mean().double()
                ssim_pbr += ssim(torch.nan_to_num(render_pkg['pbr'].clamp(0,1),nan=0.0), gt_image).mean().double()
                lpips_pbr += lpips(torch.nan_to_num(render_pkg['pbr'].clamp(0,1),nan=0.0), gt_image, net_type='vgg').mean().double()
                
                #ALBEDO IN LINEAR for psnr
                psnr_albedo += psnr(srgb_to_rgb(render_pkg['base_color']), srgb_to_rgb(gt_albedo)).mean().double()
                ssim_albedo += ssim(srgb_to_rgb(render_pkg['base_color']), srgb_to_rgb(gt_albedo)).mean().double()
                lpips_albedo += lpips(srgb_to_rgb(render_pkg['base_color']), srgb_to_rgb(gt_albedo), net_type='vgg').mean().double()
                
            
            if idx == 0:
                albedo_scale = (gt_albedo / render_pkg['base_color'].clamp(1e-6, 1))[:, mask[0] > 0].median(dim=1).values
                print("Albedo scale:", albedo_scale)

        psnr_pbr /= len(views)
        ssim_pbr /= len(views)
        lpips_pbr /= len(views)
        
        psnr_albedo /= len(views)
        ssim_albedo /= len(views)
        lpips_albedo /= len(views)
        
        
        with open(os.path.join(task_dir, f"metric.txt"), "w") as f:
            f.write(f"psnr_pbr: {psnr_pbr}\n")
            f.write(f"ssim_pbr: {ssim_pbr}\n")
            f.write(f"lpips_pbr: {lpips_pbr}\n")
            f.write(f"psnr_albedo: {psnr_albedo}\n")
            f.write(f"ssim_albedo: {ssim_albedo}\n")
            f.write(f"lpips_albedo: {lpips_albedo}\n")
            
        print("\nEvaluating {}: PSNR_PBR {} SSIM_PBR {} LPIPS_PBR {}".format(task_name, psnr_pbr, ssim_pbr, lpips_pbr))
        print("\nEvaluating {}: PSNR_ALBEDO {} SSIM_ALBEDO {} LPIPS_ALBEDO {}".format(task_name, psnr_albedo, ssim_albedo, lpips_albedo))