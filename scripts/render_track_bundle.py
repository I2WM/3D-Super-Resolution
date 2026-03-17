#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_depth_utils import encode_invdepth_u16
from depth_tooling_common import build_pipeline
from track1_bundle_utils import (
    build_bundle_camera,
    restore_gaussians_for_inference,
    validate_track1_bundle,
)


def parse_args():
    parser = ArgumentParser(description="Render the self-contained Track1 bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save_submit_rgb(output_path: Path, rendering: torch.Tensor) -> None:
    rgb = rendering.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb_u8).save(output_path, format="JPEG", quality=100)


def save_submit_depth(output_path: Path, inv_depth: torch.Tensor) -> None:
    inv_u16 = encode_invdepth_u16(inv_depth.squeeze())
    Image.fromarray(inv_u16).save(output_path)


def render_scene(scene_name: str, scene_payload: dict[str, object], output_dir: Path) -> None:
    from gaussian_renderer import render

    submit_dir = output_dir / scene_name / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)

    gaussians = restore_gaussians_for_inference(scene_payload)
    render_config = scene_payload["render_config"]
    bg_color = [1, 1, 1] if render_config["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = build_pipeline()

    with torch.no_grad():
        for camera_payload in scene_payload["test_views"]:
            camera = build_bundle_camera(scene_payload, camera_payload)
            render_pkg = render(
                camera,
                gaussians,
                pipe,
                background,
                use_trained_exp=bool(render_config["train_test_exp"]),
                separate_sh=False,
            )
            stem = Path(camera.image_name).stem
            save_submit_rgb(submit_dir / f"{stem}.JPG", render_pkg["render"])
            inv_depth = render_pkg["inv_depth"] if render_pkg.get("inv_depth") is not None else render_pkg["depth"]
            if inv_depth is None:
                raise RuntimeError(f"Rendered inv_depth is missing for {camera.image_name}")
            save_submit_depth(submit_dir / f"{stem}_depth.png", inv_depth)

    del gaussians
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(args.bundle, map_location="cpu")
    scenes = validate_track1_bundle(bundle)

    for scene_name in ("EastResearchAreas", "NorthAreas"):
        print(f"Rendering {scene_name} from bundle {args.bundle}")
        render_scene(scene_name, scenes[scene_name], args.output_dir)

    print(f"Wrote Track1 submit outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
