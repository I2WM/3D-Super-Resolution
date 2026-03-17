#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depth_tooling_common import (
    COLORMAPS,
    add_header,
    apply_colormap,
    compose_panels,
    compute_shared_range_from_paths,
    load_invdepth,
    normalize_to_u8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize submit depth with shared min/max.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing predicted *_depth.png files.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing GT *_depth.png files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save visualizations.")
    parser.add_argument("--min-value", type=float, default=None, help="Explicit shared min for pred/gt normalization.")
    parser.add_argument("--max-value", type=float, default=None, help="Explicit shared max for pred/gt normalization.")
    parser.add_argument("--lower-percentile", type=float, default=1.0, help="Percentile used to compute shared min if not set explicitly.")
    parser.add_argument("--upper-percentile", type=float, default=99.0, help="Percentile used to compute shared max if not set explicitly.")
    parser.add_argument("--diff-percentile", type=float, default=99.0, help="Percentile for abs-diff visualization upper range.")
    parser.add_argument("--colormap", choices=sorted(COLORMAPS.keys()), default="turbo")
    parser.add_argument("--label", default="", help="Optional label stored in the stats json.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(args.pred_dir.glob("*_depth.png"))
    if not pred_paths:
        raise RuntimeError(f"No *_depth.png files found in {args.pred_dir}")

    stems = [path.name for path in pred_paths]
    gt_paths = [args.gt_dir / name for name in stems]
    missing = [str(path) for path in gt_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing GT depth files: {missing[:5]}")

    if args.min_value is None or args.max_value is None:
        shared_min, shared_max = compute_shared_range_from_paths(
            pred_paths + gt_paths,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )
    else:
        shared_min = float(args.min_value)
        shared_max = float(args.max_value)

    per_image_diff_p99 = []
    pairs = []
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred = load_invdepth(pred_path)
        gt = load_invdepth(gt_path)
        valid_mask = gt > 0
        abs_diff = np.abs(pred - gt)
        if np.any(valid_mask):
            per_image_diff_p99.append(float(np.percentile(abs_diff[valid_mask], args.diff_percentile)))
        pairs.append((pred_path, gt_path, pred, gt, valid_mask, abs_diff))

    diff_max = max(per_image_diff_p99) if per_image_diff_p99 else 1e-6
    diff_max = max(diff_max, 1e-6)

    per_view = []
    for pred_path, gt_path, pred, gt, valid_mask, abs_diff in pairs:
        pred_u8 = normalize_to_u8(pred, shared_min, shared_max, valid_mask)
        gt_u8 = normalize_to_u8(gt, shared_min, shared_max, valid_mask)
        diff_u8 = normalize_to_u8(abs_diff, 0.0, diff_max, valid_mask)

        pred_vis = add_header(apply_colormap(pred_u8, valid_mask, args.colormap), "Pred")
        gt_vis = add_header(apply_colormap(gt_u8, valid_mask, args.colormap), "GT")
        diff_vis = add_header(apply_colormap(diff_u8, valid_mask, args.colormap), "Abs Diff")

        canvas = compose_panels(
            [pred_vis, gt_vis, diff_vis],
            footer_lines=[
                f"shared range [{shared_min:.6f}, {shared_max:.6f}]  diff range [0.000000, {diff_max:.6f}]  {pred_path.name}"
            ],
        )
        output_path = args.output_dir / pred_path.name.replace("_depth.png", "_viz.png")
        cv2.imwrite(str(output_path), canvas)

        valid_count = int(valid_mask.sum())
        per_view.append(
            {
                "file": pred_path.name,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "output_path": str(output_path),
                "valid_pixels": valid_count,
                "mean_abs_diff": float(abs_diff[valid_mask].mean()) if valid_count else None,
                "max_abs_diff": float(abs_diff[valid_mask].max()) if valid_count else None,
            }
        )

    stats = {
        "label": args.label,
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "output_dir": str(args.output_dir),
        "shared_min": shared_min,
        "shared_max": shared_max,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "diff_percentile": args.diff_percentile,
        "diff_max": diff_max,
        "colormap": args.colormap,
        "num_views": len(per_view),
        "per_view": per_view,
    }
    (args.output_dir / "range_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"shared_min": shared_min, "shared_max": shared_max, "diff_max": diff_max, "num_views": len(per_view)}, indent=2))


if __name__ == "__main__":
    main()
