from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


INVDEPTH_EXPORT_SCALE = 255.0 * 256.0
PNG_MIN_INVDEPTH = 0.0
UINT16_MAX_INVDEPTH = np.iinfo(np.uint16).max / INVDEPTH_EXPORT_SCALE


@dataclass(frozen=True)
class DepthAdjustmentSpec:
    config_path: str
    fit_granularity: str = "scene"
    fit_mode: str = "scale_only"
    default_scale: float = 1.0
    default_bias: float = 0.0
    per_image: dict[str, tuple[float, float]] = field(default_factory=dict)

    def resolve(self, image_name: str) -> tuple[float, float]:
        path = Path(image_name)
        candidates = [
            str(image_name),
            path.name,
            path.stem,
            f"{path.stem}_depth.png",
        ]
        for candidate in candidates:
            if candidate in self.per_image:
                return self.per_image[candidate]
        return self.default_scale, self.default_bias

    @property
    def num_entries(self) -> int:
        return len(self.per_image)


def decode_invdepth_u16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) / INVDEPTH_EXPORT_SCALE


def clip_png_invdepth(values):
    if torch is not None and isinstance(values, torch.Tensor):
        return values.clamp(min=PNG_MIN_INVDEPTH, max=UINT16_MAX_INVDEPTH)
    return np.clip(values, PNG_MIN_INVDEPTH, UINT16_MAX_INVDEPTH)


def apply_invdepth_affine(values, scale: float = 1.0, bias: float = 0.0):
    if scale == 1.0 and bias == 0.0:
        return values
    return values * scale + bias


def encode_invdepth_u16(values) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    clipped = clip_png_invdepth(values)
    return np.clip(
        np.rint(clipped.astype(np.float32) * INVDEPTH_EXPORT_SCALE),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _normalize_fit_entry(entry: dict[str, Any], default_scale: float, default_bias: float) -> tuple[str | None, tuple[float, float]]:
    key = entry.get("file") or entry.get("image_name") or entry.get("filename") or entry.get("name")
    if key is None:
        return None, (default_scale, default_bias)
    scale = _to_float(entry.get("scale_a"), default_scale)
    bias = _to_float(entry.get("bias_b"), default_bias)
    return Path(str(key)).name, (scale, bias)


def resolve_depth_adjustment_path(path_or_dir: str | Path | None) -> Path | None:
    if path_or_dir is None:
        return None
    text = str(path_or_dir).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_dir():
        for name in ("depth_adjustment.json", "scale_fit.json", "per_view_fits.json"):
            probe = candidate / name
            if probe.exists():
                return probe.resolve()
        raise FileNotFoundError(
            f"Could not find depth adjustment json under directory {candidate}. "
            "Expected one of depth_adjustment.json, scale_fit.json, per_view_fits.json."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"Depth adjustment config does not exist: {candidate}")
    return candidate.resolve()


def load_depth_adjustment_spec(path_or_dir: str | Path | None) -> DepthAdjustmentSpec | None:
    resolved = resolve_depth_adjustment_path(path_or_dir)
    if resolved is None:
        return None

    payload = json.loads(resolved.read_text())
    if isinstance(payload, list):
        fit_granularity = "image"
        fit_mode = "custom"
        default_scale = 1.0
        default_bias = 0.0
        entries = payload
    else:
        fit_granularity = str(payload.get("fit_granularity", "scene"))
        fit_mode = str(payload.get("fit_mode", "scale_only"))
        default_scale = _to_float(payload.get("scale_a"), 1.0)
        default_bias = _to_float(payload.get("bias_b"), 0.0)
        entries = payload.get("per_view_fits", [])

    per_image: dict[str, tuple[float, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key, values = _normalize_fit_entry(entry, default_scale, default_bias)
        if key is None:
            continue
        per_image[key] = values

    return DepthAdjustmentSpec(
        config_path=str(resolved),
        fit_granularity=fit_granularity,
        fit_mode=fit_mode,
        default_scale=default_scale,
        default_bias=default_bias,
        per_image=per_image,
    )
