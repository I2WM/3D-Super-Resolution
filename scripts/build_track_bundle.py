#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track1_bundle_utils import build_track1_bundle_payload


def parse_args():
    parser = ArgumentParser(description="Build the self-contained Track1 render bundle.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    bundle, summaries = build_track1_bundle_payload()
    for summary in summaries:
        print(f"[{summary['scene_name']}] checkpoint: {summary['checkpoint_path']}")
        print(f"[{summary['scene_name']}] iteration=40000")
        print(f"[{summary['scene_name']}] num_points={summary['num_points']}")
        print(f"[{summary['scene_name']}] affine={summary['fit_signature']}")
        print(f"[{summary['scene_name']}] num_test_views={summary['num_test_views']}")

    torch.save(bundle, args.output)
    print(f"Wrote Track1 bundle: {args.output}")


if __name__ == "__main__":
    main()
