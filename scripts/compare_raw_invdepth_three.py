#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_depth_utils import UINT16_MAX_INVDEPTH
from depth_tooling_common import (
    COLORMAPS,
    add_header,
    apply_colormap,
    compose_panels,
    compute_shared_range,
    normalize_to_u8,
    render_test_invdepths,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw float inv_depth from two models against GT.")
    parser.add_argument("--left-model", type=Path, required=True)
    parser.add_argument("--middle-model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--middle-label", default="middle")
    parser.add_argument("--right-label", default="GT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colormap", choices=sorted(COLORMAPS.keys()), default="turbo")
    parser.add_argument("--min-value", type=float, default=None)
    parser.add_argument("--max-value", type=float, default=None)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--label", default="")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_npz"
    raw_dir.mkdir(parents=True, exist_ok=True)

    left_pred, left_gt, left_cfg = render_test_invdepths(args.left_model, args.iteration)
    middle_pred, middle_gt, middle_cfg = render_test_invdepths(args.middle_model, args.iteration)

    if left_cfg.source_path != middle_cfg.source_path:
        raise RuntimeError("Left and middle models do not share the same source_path.")

    names = sorted(left_pred.keys())
    if names != sorted(middle_pred.keys()):
        raise RuntimeError("Left and middle models do not have matching test views.")

    gt_map = left_gt
    if args.min_value is None or args.max_value is None:
        shared_min, shared_max = compute_shared_range(
            [left_pred[name] for name in names] + [middle_pred[name] for name in names] + [gt_map[name] for name in names],
            args.lower_percentile,
            args.upper_percentile,
        )
    else:
        shared_min = float(args.min_value)
        shared_max = float(args.max_value)

    per_view = []
    for name in names:
        left = left_pred[name]
        middle = middle_pred[name]
        gt = gt_map[name]
        mask = gt > 0

        np.savez_compressed(raw_dir / f"{Path(name).stem}_raw.npz", left=left, middle=middle, gt=gt)

        left_vis = add_header(
            apply_colormap(normalize_to_u8(left, shared_min, shared_max, mask), mask, args.colormap),
            args.left_label,
        )
        middle_vis = add_header(
            apply_colormap(normalize_to_u8(middle, shared_min, shared_max, mask), mask, args.colormap),
            args.middle_label,
        )
        right_vis = add_header(
            apply_colormap(normalize_to_u8(gt, shared_min, shared_max, mask), mask, args.colormap),
            args.right_label,
        )
        left_sat = float((left[mask] > UINT16_MAX_INVDEPTH).mean()) if np.any(mask) else 0.0
        middle_sat = float((middle[mask] > UINT16_MAX_INVDEPTH).mean()) if np.any(mask) else 0.0
        canvas = compose_panels(
            [left_vis, middle_vis, right_vis],
            footer_lines=[
                f"shared raw range [{shared_min:.6f}, {shared_max:.6f}]  file {name}",
                f"uint16 clip ratio  left={left_sat:.4f}  middle={middle_sat:.4f}  clip_thr={UINT16_MAX_INVDEPTH:.6f}",
            ],
        )
        output_path = args.output_dir / f"{Path(name).stem}_raw_compare.png"
        cv2.imwrite(str(output_path), canvas)

        valid_count = int(mask.sum())
        per_view.append(
            {
                "file": name,
                "output_path": str(output_path),
                "raw_npz": str(raw_dir / f"{Path(name).stem}_raw.npz"),
                "valid_pixels": valid_count,
                "left_mean": float(left[mask].mean()) if valid_count else None,
                "middle_mean": float(middle[mask].mean()) if valid_count else None,
                "gt_mean": float(gt[mask].mean()) if valid_count else None,
                "left_clip_ratio": left_sat,
                "middle_clip_ratio": middle_sat,
            }
        )

    stats = {
        "label": args.label,
        "left_model": str(args.left_model),
        "middle_model": str(args.middle_model),
        "source_path": left_cfg.source_path,
        "output_dir": str(args.output_dir),
        "iteration": args.iteration,
        "shared_min": shared_min,
        "shared_max": shared_max,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "colormap": args.colormap,
        "uint16_clip_threshold": UINT16_MAX_INVDEPTH,
        "num_views": len(per_view),
        "per_view": per_view,
    }
    (args.output_dir / "range_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({
        "shared_min": shared_min,
        "shared_max": shared_max,
        "num_views": len(per_view),
        "clip_threshold": UINT16_MAX_INVDEPTH,
    }, indent=2))


if __name__ == "__main__":
    main()
