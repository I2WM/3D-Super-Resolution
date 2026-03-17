#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_depth_utils import decode_invdepth_u16


COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "gray": None,
}


def load_cfg_args(model_path: Path) -> SimpleNamespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args at {cfg_path}")
    text = cfg_path.read_text().strip()
    expr = ast.parse(text, mode="eval").body
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
    kwargs["source_path"] = str(Path(kwargs["source_path"]).resolve())
    kwargs["model_path"] = str(model_path.resolve())
    return SimpleNamespace(**kwargs)


def build_pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        debug=False,
        antialiasing=False,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    )


def render_test_invdepths(
    model_path: Path, iteration: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], SimpleNamespace]:
    import torch

    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    cfg = load_cfg_args(model_path)
    gaussians = GaussianModel(cfg.sh_degree)
    scene = Scene(cfg, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = build_pipeline()

    pred = {}
    gt = {}
    with torch.no_grad():
        for view in scene.getTestCameras():
            pkg = render(
                view,
                gaussians,
                pipe,
                background,
                use_trained_exp=getattr(cfg, "train_test_exp", False),
                separate_sh=False,
            )
            raw_inv_depth = pkg.get("raw_inv_depth")
            inv_depth = raw_inv_depth if raw_inv_depth is not None else pkg.get("inv_depth")
            if inv_depth is None:
                inv_depth = pkg.get("depth")
            if inv_depth is None:
                raise RuntimeError(f"Rendered inv_depth is missing for {view.image_name}")
            if view.invdepthmap is None:
                raise RuntimeError(f"GT invdepthmap is missing for {view.image_name}")
            pred[view.image_name] = inv_depth.squeeze().detach().cpu().numpy().astype(np.float32)
            gt[view.image_name] = view.invdepthmap.squeeze().detach().cpu().numpy().astype(np.float32)

    if not pred:
        raise RuntimeError(f"No test views found for {model_path}")
    return pred, gt, cfg


def load_u16_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read {path}")
    return image.astype(np.uint16, copy=False)


def load_invdepth(path: Path) -> np.ndarray:
    return decode_invdepth_u16(load_u16_png(path))


def compute_shared_range(
    arrays: list[np.ndarray], lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
    valid_values = []
    for values in arrays:
        mask = values > 0
        if np.any(mask):
            valid_values.append(values[mask])
    if not valid_values:
        raise RuntimeError("No valid depth pixels found for shared min/max.")
    merged = np.concatenate(valid_values)
    min_value = float(np.percentile(merged, lower_percentile))
    max_value = float(np.percentile(merged, upper_percentile))
    if max_value <= min_value:
        max_value = min_value + 1e-6
    return min_value, max_value


def compute_shared_range_from_paths(
    paths: list[Path], lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
    return compute_shared_range(
        [load_invdepth(path) for path in paths],
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def normalize_to_u8(values: np.ndarray, min_value: float, max_value: float, mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(values, dtype=np.float32)
    scale = max(max_value - min_value, 1e-6)
    normalized[mask] = np.clip((values[mask] - min_value) / scale, 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def apply_colormap(gray_u8: np.ndarray, mask: np.ndarray, colormap_name: str) -> np.ndarray:
    if colormap_name == "gray":
        colored = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    else:
        colored = cv2.applyColorMap(gray_u8, COLORMAPS[colormap_name])
    colored[~mask] = 0
    return colored


def add_header(image: np.ndarray, title: str) -> np.ndarray:
    header_h = 40
    canvas = np.zeros((image.shape[0] + header_h, image.shape[1], 3), dtype=np.uint8)
    canvas[header_h:] = image
    cv2.putText(
        canvas,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_panels(panels: list[np.ndarray], footer_lines: list[str] | tuple[str, ...] = ()) -> np.ndarray:
    combined = np.concatenate(panels, axis=1)
    lines = [line for line in footer_lines if line]
    if not lines:
        return combined

    footer_h = 16 + 30 * len(lines)
    canvas = np.zeros((combined.shape[0] + footer_h, combined.shape[1], 3), dtype=np.uint8)
    canvas[:combined.shape[0]] = combined
    for index, line in enumerate(lines):
        y = combined.shape[0] + 24 + index * 30
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas
