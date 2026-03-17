#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_utils import TESTVIEW_IDS
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


TRACK1_BUNDLE_TYPE = "track1_render_bundle"
TRACK1_BUNDLE_VERSION = 1
TRACK1_RENDER_CONFIG = {
    "sh_degree": 3,
    "white_background": False,
    "competition_mode": True,
    "train_test_exp": False,
}
TRACK1_SCENE_SPECS = {
    "EastResearchAreas": {
        "checkpoint_path": Path(
            "/proj-tango-pvc/users/liujun/3dsr/cgj-gaussian-splatting/workdirs/competition/tuning/track1/"
            "dualscene_off_l0p1_20260316/east_from30k_p2_10000_off_l0p1/EastResearchAreas/chkpnt40000.pth"
        ),
        "scale_fit_path": Path(
            "/proj-tango-pvc/users/liujun/3dsr/cgj-gaussian-splatting/workdirs/competition/analysis/"
            "track1_affine_refit_20260318/east_off_l0p1_image_png_l1/scale_fit.json"
        ),
        "per_view_fits_path": Path(
            "/proj-tango-pvc/users/liujun/3dsr/cgj-gaussian-splatting/workdirs/competition/analysis/"
            "track1_affine_refit_20260318/east_off_l0p1_image_png_l1/per_view_fits.json"
        ),
    },
    "NorthAreas": {
        "checkpoint_path": Path(
            "/proj-tango-pvc/users/liujun/3dsr/cgj-gaussian-splatting/workdirs/competition/tuning/track1/"
            "dualscene_off_l0p1_20260316/north_from30k_p2_10000_off_l0p1/NorthAreas/chkpnt40000.pth"
        ),
        "scale_fit_path": Path(
            "/proj-tango-pvc/users/liujun/3dsr/cgj-gaussian-splatting/workdirs/competition/analysis/"
            "track1_affine_refit_20260318/north_off_l0p1_image_png_l1/scale_fit.json"
        ),
        "per_view_fits_path": Path(
            "/proj-tango-pvc/users/liujun/3dsr/cgj-gaussian-splatting/workdirs/competition/analysis/"
            "track1_affine_refit_20260318/north_off_l0p1_image_png_l1/per_view_fits.json"
        ),
    },
}
def parse_cfg_args(model_dir: Path) -> dict[str, object]:
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args at {cfg_path}")
    text = cfg_path.read_text().strip()
    expr = ast.parse(text, mode="eval").body
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
    return kwargs


def validate_cfg_args(scene_name: str, model_dir: Path, cfg: dict[str, object]) -> None:
    required = {
        "competition_mode": True,
        "train_test_exp": False,
        "white_background": False,
        "sh_degree": 3,
    }
    for key, expected in required.items():
        actual = cfg.get(key)
        if actual != expected:
            raise RuntimeError(
                f"{scene_name} cfg_args validation failed for {model_dir}: "
                f"{key}={actual!r}, expected {expected!r}"
            )


def resolve_render_resolution(source_image_path: Path, resolution_arg: object, resolution_scale: float = 1.0) -> tuple[int, int]:
    with Image.open(source_image_path) as image:
        orig_w, orig_h = image.size

    resolution = int(resolution_arg)
    if resolution in (1, 2, 4, 8):
        return (
            round(orig_w / (resolution_scale * resolution)),
            round(orig_h / (resolution_scale * resolution)),
        )

    if resolution == -1:
        global_down = (orig_w / 1600.0) if orig_w > 1600 else 1.0
    else:
        global_down = orig_w / float(resolution)
    scale = float(global_down) * float(resolution_scale)
    return (int(orig_w / scale), int(orig_h / scale))


def load_trusted_checkpoint(checkpoint_path: Path) -> tuple[tuple[object, ...], int]:
    if checkpoint_path.name != "chkpnt40000.pth":
        raise RuntimeError(f"Unexpected checkpoint filename: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError(f"Unexpected checkpoint payload structure in {checkpoint_path}")
    model_args, iteration = payload
    if int(iteration) != 40000:
        raise RuntimeError(f"{checkpoint_path} has iteration {iteration}, expected 40000")
    if not isinstance(model_args, tuple) or len(model_args) != 12:
        raise RuntimeError(f"Unexpected model_args structure in {checkpoint_path}")
    return model_args, int(iteration)


def extract_gaussian_state(model_args: tuple[object, ...]) -> dict[str, object]:
    return {
        "active_sh_degree": int(model_args[0]),
        "xyz": model_args[1].detach().cpu().contiguous(),
        "features_dc": model_args[2].detach().cpu().contiguous(),
        "features_rest": model_args[3].detach().cpu().contiguous(),
        "scaling": model_args[4].detach().cpu().contiguous(),
        "rotation": model_args[5].detach().cpu().contiguous(),
        "opacity": model_args[6].detach().cpu().contiguous(),
        "spatial_lr_scale": float(model_args[11]),
    }


def load_scale_fit(scale_fit_path: Path) -> dict[str, object]:
    payload = json.loads(scale_fit_path.read_text())
    required = {
        "fit_mode": "affine",
        "fit_granularity": "image",
        "fit_loss": "l1",
        "fit_space": "png",
    }
    for key, expected in required.items():
        actual = payload.get(key)
        if actual != expected:
            raise RuntimeError(
                f"{scale_fit_path} validation failed: {key}={actual!r}, expected {expected!r}"
            )
    return payload


def load_per_view_affine(
    per_view_fits_path: Path,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    payload = json.loads(per_view_fits_path.read_text())
    if not isinstance(payload, list) or len(payload) != len(TESTVIEW_IDS):
        raise RuntimeError(
            f"{per_view_fits_path} must contain exactly {len(TESTVIEW_IDS)} entries; got {len(payload)}"
        )
    per_image: dict[str, dict[str, float]] = {}
    ordered_names: list[str] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid per-view entry in {per_view_fits_path}: {entry!r}")
        image_name = str(entry["file"])
        ordered_names.append(image_name)
        per_image[image_name] = {
            "scale_a": float(entry["scale_a"]),
            "bias_b": float(entry["bias_b"]),
        }
    return per_image, ordered_names


def _camera_payload_from_json_row(row: dict[str, object]) -> dict[str, object]:
    rotation = np.asarray(row["rotation"], dtype=np.float32)
    position = np.asarray(row["position"], dtype=np.float32)
    translation = -(rotation.T @ position)
    return {
        "uid": int(row["id"]),
        "colmap_id": int(row["id"]),
        "image_name": str(row["img_name"]),
        "R": rotation.tolist(),
        "T": translation.tolist(),
        "FoVx": float(focal2fov(float(row["fx"]), int(row["width"]))),
        "FoVy": float(focal2fov(float(row["fy"]), int(row["height"]))),
    }


def load_track1_test_views(
    model_dir: Path,
    scene_name: str,
    ordered_names: list[str],
    cfg: dict[str, object],
) -> list[dict[str, object]]:
    cameras_json_path = model_dir / "cameras.json"
    if not cameras_json_path.exists():
        raise FileNotFoundError(f"Missing cameras.json at {cameras_json_path}")
    source_images_dir = Path(str(cfg["source_path"])) / str(cfg.get("images", "images"))
    resolution_arg = cfg.get("resolution", 1)

    rows = json.loads(cameras_json_path.read_text())
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid cameras.json payload in {cameras_json_path}")

    target_names = set(ordered_names)
    first_match_by_name: dict[str, dict[str, object]] = {}
    for row in rows:
        image_name = str(row.get("img_name"))
        if image_name in target_names and image_name not in first_match_by_name:
            first_match_by_name[image_name] = _camera_payload_from_json_row(row)
            if len(first_match_by_name) == len(target_names):
                break

    if len(first_match_by_name) != len(target_names):
        missing = sorted(target_names - set(first_match_by_name.keys()))
        raise RuntimeError(
            f"{scene_name} cameras.json is missing Track1 test views: {missing}"
        )

    test_views = []
    for name in ordered_names:
        image_path = source_images_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing Track1 source image for bundled test view: {image_path}")
        width, height = resolve_render_resolution(image_path, resolution_arg=resolution_arg)
        payload = dict(first_match_by_name[name])
        payload["width"] = int(width)
        payload["height"] = int(height)
        test_views.append(payload)
    if len(test_views) != len(TESTVIEW_IDS):
        raise RuntimeError(
            f"{scene_name} produced {len(test_views)} test views, expected {len(TESTVIEW_IDS)}"
        )
    return test_views


def validate_test_view_names(scene_name: str, test_views: list[dict[str, object]], per_image: dict[str, dict[str, float]]) -> None:
    camera_names = {str(view["image_name"]) for view in test_views}
    affine_names = set(per_image.keys())
    if camera_names != affine_names:
        raise RuntimeError(
            f"{scene_name} test-view names do not match affine names.\n"
            f"Only in cameras: {sorted(camera_names - affine_names)}\n"
            f"Only in affine: {sorted(affine_names - camera_names)}"
        )


def build_track1_scene_bundle(scene_name: str) -> tuple[dict[str, object], dict[str, object]]:
    spec = TRACK1_SCENE_SPECS[scene_name]
    checkpoint_path = spec["checkpoint_path"]
    scale_fit_path = spec["scale_fit_path"]
    per_view_fits_path = spec["per_view_fits_path"]

    for path in (checkpoint_path, scale_fit_path, per_view_fits_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Track1 bundle input: {path}")

    model_dir = checkpoint_path.parent
    cfg = parse_cfg_args(model_dir)
    validate_cfg_args(scene_name, model_dir, cfg)
    model_args, iteration = load_trusted_checkpoint(checkpoint_path)
    gaussian_state = extract_gaussian_state(model_args)
    scale_fit = load_scale_fit(scale_fit_path)
    per_image, ordered_names = load_per_view_affine(per_view_fits_path)
    test_views = load_track1_test_views(model_dir, scene_name, ordered_names, cfg)
    validate_test_view_names(scene_name, test_views, per_image)

    scene_bundle = {
        "scene_name": scene_name,
        "iteration": iteration,
        "render_config": dict(TRACK1_RENDER_CONFIG),
        "gaussians": gaussian_state,
        "test_views": test_views,
        "depth_adjustment": {
            "fit_mode": str(scale_fit["fit_mode"]),
            "fit_granularity": str(scale_fit["fit_granularity"]),
            "fit_loss": str(scale_fit["fit_loss"]),
            "fit_space": str(scale_fit["fit_space"]),
            "per_image": per_image,
        },
    }
    summary = {
        "scene_name": scene_name,
        "checkpoint_path": str(checkpoint_path),
        "scale_fit_path": str(scale_fit_path),
        "per_view_fits_path": str(per_view_fits_path),
        "iteration": iteration,
        "num_points": int(gaussian_state["xyz"].shape[0]),
        "num_test_views": len(test_views),
        "fit_signature": "image + affine + l1 + png",
    }
    return scene_bundle, summary


def build_track1_bundle_payload() -> tuple[dict[str, object], list[dict[str, object]]]:
    scenes: dict[str, object] = {}
    summaries: list[dict[str, object]] = []
    for scene_name in ("EastResearchAreas", "NorthAreas"):
        scene_bundle, summary = build_track1_scene_bundle(scene_name)
        scenes[scene_name] = scene_bundle
        summaries.append(summary)
    bundle = {
        "bundle_type": TRACK1_BUNDLE_TYPE,
        "bundle_version": TRACK1_BUNDLE_VERSION,
        "track": "track1",
        "scenes": scenes,
    }
    return bundle, summaries


def validate_track1_bundle(bundle: dict[str, object]) -> dict[str, object]:
    if bundle.get("bundle_type") != TRACK1_BUNDLE_TYPE:
        raise RuntimeError(f"Unexpected bundle_type: {bundle.get('bundle_type')!r}")
    if int(bundle.get("bundle_version", -1)) != TRACK1_BUNDLE_VERSION:
        raise RuntimeError(f"Unexpected bundle_version: {bundle.get('bundle_version')!r}")
    if bundle.get("track") != "track1":
        raise RuntimeError(f"Unexpected track in bundle: {bundle.get('track')!r}")
    scenes = bundle.get("scenes")
    if not isinstance(scenes, dict):
        raise RuntimeError("Bundle scenes payload is missing or invalid.")
    for scene_name in ("EastResearchAreas", "NorthAreas"):
        if scene_name not in scenes:
            raise RuntimeError(f"Bundle is missing scene {scene_name}")
        scene_payload = scenes[scene_name]
        if int(scene_payload.get("iteration", -1)) != 40000:
            raise RuntimeError(f"{scene_name} bundle iteration is not 40000")
        test_views = scene_payload.get("test_views")
        if not isinstance(test_views, list) or len(test_views) != len(TESTVIEW_IDS):
            raise RuntimeError(f"{scene_name} bundle must contain exactly {len(TESTVIEW_IDS)} test views")
        depth_adjustment = scene_payload.get("depth_adjustment", {})
        per_image = depth_adjustment.get("per_image")
        if not isinstance(per_image, dict) or len(per_image) != len(TESTVIEW_IDS):
            raise RuntimeError(f"{scene_name} bundle must contain per-image affine parameters for all test views")
    return scenes


def restore_gaussians_for_inference(scene_payload: dict[str, object]):
    from gaussian_renderer import GaussianModel

    sh_degree = int(scene_payload["render_config"]["sh_degree"])
    gaussians = GaussianModel(sh_degree)
    gaussians.restore_inference_only(scene_payload["gaussians"])
    return gaussians


class Track1BundleCamera:
    def __init__(self, camera_payload: dict[str, object], scale: float, bias: float):
        self.uid = int(camera_payload["uid"])
        self.colmap_id = int(camera_payload["colmap_id"])
        self.image_name = str(camera_payload["image_name"])
        self.image_width = int(camera_payload["width"])
        self.image_height = int(camera_payload["height"])
        self.FoVx = float(camera_payload["FoVx"])
        self.FoVy = float(camera_payload["FoVy"])
        self.znear = 0.01
        self.zfar = 100.0
        self.depth_adjustment_scale = float(scale)
        self.depth_adjustment_bias = float(bias)
        self.depth_adjustment_source = TRACK1_BUNDLE_TYPE

        R = np.asarray(camera_payload["R"], dtype=np.float32)
        T = np.asarray(camera_payload["T"], dtype=np.float32)
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def build_bundle_camera(scene_payload: dict[str, object], camera_payload: dict[str, object]) -> Track1BundleCamera:
    per_image = scene_payload["depth_adjustment"]["per_image"]
    params = per_image[str(camera_payload["image_name"])]
    return Track1BundleCamera(
        camera_payload=camera_payload,
        scale=float(params["scale_a"]),
        bias=float(params["bias_b"]),
    )
