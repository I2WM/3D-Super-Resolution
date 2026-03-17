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
    parser = argparse.ArgumentParser(description="Visualize three depth directories with one shared min/max.")
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--middle-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-label", default="Left")
    parser.add_argument("--middle-label", default="Middle")
    parser.add_argument("--right-label", default="Right")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-value", type=float, default=None)
    parser.add_argument("--max-value", type=float, default=None)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--colormap", choices=sorted(COLORMAPS.keys()), default="gray")
    parser.add_argument("--label", default="")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left_paths = sorted(args.left_dir.glob("*_depth.png"))
    if not left_paths:
        raise RuntimeError(f"No *_depth.png files found in {args.left_dir}")
    names = [p.name for p in left_paths]
    middle_paths = [args.middle_dir / name for name in names]
    right_paths = [args.right_dir / name for name in names]
    missing = [str(p) for p in middle_paths + right_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing files: {missing[:5]}")

    if args.min_value is None or args.max_value is None:
        shared_min, shared_max = compute_shared_range_from_paths(
            left_paths + middle_paths + right_paths,
            args.lower_percentile,
            args.upper_percentile,
        )
    else:
        shared_min = float(args.min_value)
        shared_max = float(args.max_value)

    per_view = []
    for left_path, middle_path, right_path in zip(left_paths, middle_paths, right_paths):
        left = load_invdepth(left_path)
        middle = load_invdepth(middle_path)
        right = load_invdepth(right_path)
        mask = (left > 0) | (middle > 0) | (right > 0)

        left_vis = add_header(
            apply_colormap(normalize_to_u8(left, shared_min, shared_max, mask), mask, args.colormap),
            args.left_label,
        )
        middle_vis = add_header(
            apply_colormap(normalize_to_u8(middle, shared_min, shared_max, mask), mask, args.colormap),
            args.middle_label,
        )
        right_vis = add_header(
            apply_colormap(normalize_to_u8(right, shared_min, shared_max, mask), mask, args.colormap),
            args.right_label,
        )

        canvas = compose_panels(
            [left_vis, middle_vis, right_vis],
            footer_lines=[f"shared range [{shared_min:.6f}, {shared_max:.6f}]  {left_path.name}"],
        )
        output_path = args.output_dir / left_path.name.replace("_depth.png", "_compare.png")
        cv2.imwrite(str(output_path), canvas)
        per_view.append(
            {
                "file": left_path.name,
                "left_path": str(left_path),
                "middle_path": str(middle_path),
                "right_path": str(right_path),
                "output_path": str(output_path),
            }
        )

    stats = {
        "label": args.label,
        "left_dir": str(args.left_dir),
        "middle_dir": str(args.middle_dir),
        "right_dir": str(args.right_dir),
        "output_dir": str(args.output_dir),
        "shared_min": shared_min,
        "shared_max": shared_max,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "colormap": args.colormap,
        "num_views": len(per_view),
        "per_view": per_view,
    }
    (args.output_dir / "range_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"shared_min": shared_min, "shared_max": shared_max, "num_views": len(per_view)}, indent=2))


if __name__ == "__main__":
    main()
