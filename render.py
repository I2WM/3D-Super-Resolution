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

import os
import shutil
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from competition_depth_utils import encode_invdepth_u16
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from utils.general_utils import safe_state
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def save_competition_submission(model_path, view, render_pkg, source_path=None):
    submit_path = Path(model_path) / "submit"
    submit_gt_path = Path(model_path) / "submit_gt"
    submit_path.mkdir(parents=True, exist_ok=True)
    submit_gt_path.mkdir(parents=True, exist_ok=True)

    stem = Path(view.image_name).stem
    rgb = render_pkg["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb_u8).save(submit_path / f"{stem}.JPG", format="JPEG", quality=100)

    gt_target = submit_gt_path / f"{stem}.JPG"
    gt_source = None
    if source_path is not None:
        candidate = Path(source_path) / "images" / f"{stem}.JPG"
        if candidate.exists():
            gt_source = candidate
    if gt_source is not None:
        shutil.copy2(gt_source, gt_target)
    else:
        gt = view.original_image[0:3, :, :].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        gt_u8 = np.clip(np.rint(gt * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(gt_u8).save(gt_target, format="JPEG", quality=100)

    gt_depth_target = submit_gt_path / f"{stem}_depth.png"
    gt_depth_source = None
    if source_path is not None:
        candidate = Path(source_path) / "depth" / f"{stem}_depth.png"
        if candidate.exists():
            gt_depth_source = candidate
    if gt_depth_source is not None:
        shutil.copy2(gt_depth_source, gt_depth_target)
    elif getattr(view, "invdepthmap", None) is not None:
        gt_inv_u16 = encode_invdepth_u16(view.invdepthmap.squeeze())
        Image.fromarray(gt_inv_u16).save(gt_depth_target)

    inv_depth = render_pkg["inv_depth"] if render_pkg.get("inv_depth") is not None else render_pkg["depth"]
    inv_u16 = encode_invdepth_u16(inv_depth.squeeze())
    Image.fromarray(inv_u16).save(submit_path / f"{stem}_depth.png")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, competition_mode, source_path=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if competition_mode and name == "test":
            save_competition_submission(model_path, view, render_pkg, source_path=source_path)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        competition_mode = getattr(dataset, "competition_mode", False)
        if competition_mode:
            skip_train = True

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, competition_mode, source_path=dataset.source_path)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, competition_mode, source_path=dataset.source_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
