#!/usr/bin/env python3

import ast
import csv
import importlib.util
import json
import re
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_depth_utils import decode_invdepth_u16
from competition_utils import TESTVIEW_IDS
from depth_tooling_common import load_u16_png

DEFAULT_SCENES = ["EastResearchAreas", "NorthAreas"]
DEPTH_EPE_RE = re.compile(r"PSNR=([0-9.]+), Depth EPE=([0-9.]+), Score=([0-9.]+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_colmap_loader():
    loader_path = repo_root() / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("colmap_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_prediction_root() -> Path:
    return repo_root() / "workdirs" / "competition" / "exp" / "track1"


def default_proxy_gt_root() -> Path:
    return Path("/tmp/scratch-space/ntire2026/3dsr/test_trainset/3dsr_track1")


def default_log_path() -> Path:
    return repo_root() / "test_track1_log.txt"


def default_scene_root() -> Path:
    return Path("/tmp/scratch-space/ntire2026/3dsr/test_trainset/3dsr_track1")


def default_output_dir(label: str) -> Path:
    return repo_root() / "workdirs" / "competition" / "analysis" / label


def parse_submission_log(log_path: Path):
    entries = {}
    lines = [line.strip() for line in log_path.read_text().splitlines() if line.strip()]

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("[") and "_depth.png" in line:
            depth_paths = [Path(path) for path in ast.literal_eval(line)]
            if not depth_paths:
                idx += 1
                continue
            scene = depth_paths[0].parent.name
            filenames = [path.name for path in depth_paths]
            official = []
            probe = idx + 1
            while probe < len(lines):
                candidate = lines[probe]
                if candidate.startswith("[") and "_depth.png" in candidate:
                    break
                match = DEPTH_EPE_RE.search(candidate)
                if match:
                    official.append(
                        {
                            "psnr": float(match.group(1)),
                            "depth_epe": float(match.group(2)),
                            "score": float(match.group(3)),
                        }
                    )
                probe += 1

            if len(official) != len(filenames):
                raise RuntimeError(
                    f"Submission log scene {scene} has {len(filenames)} depth files but {len(official)} metric rows."
                )
            entries[scene] = {"filenames": filenames, "metrics": official}
            idx = probe
            continue
        idx += 1

    missing = [scene for scene in DEFAULT_SCENES if scene not in entries]
    if missing:
        raise RuntimeError(f"Submission log is missing scenes: {missing}")
    return entries


def expected_test_depth_filenames(scene_root: Path, scene: str):
    colmap_loader = load_colmap_loader()
    extrinsics = colmap_loader.read_extrinsics_binary(str(scene_root / scene / "sparse" / "0" / "images.bin"))
    image_names = [extrinsics[key].name for key in extrinsics]
    stems = [Path(image_names[idx]).stem for idx in TESTVIEW_IDS]
    return [f"{stem}_depth.png" for stem in stems]


def resolve_prediction_depth_dir(pred_root: Path, scene: str, expected_filenames):
    candidates = [
        pred_root / scene / f"{scene}_submit",
        pred_root / scene / "submit",
        pred_root / scene / "depth",
        pred_root / scene,
    ]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / filename).exists() for filename in expected_filenames):
            return candidate
    raise FileNotFoundError(
        f"Could not find a prediction depth directory for {scene} under {pred_root} that contains all expected test depth files."
    )

def safe_pearson(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def safe_spearman(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return safe_pearson(rankdata(x), rankdata(y))


def masked_metrics(pred_u16, gt_u16):
    mask = gt_u16 > 0
    if not mask.any():
        raise RuntimeError("Proxy GT depth contains no valid pixels.")
    pred = pred_u16.astype(np.float64)
    gt = gt_u16.astype(np.float64)
    pred_inv = decode_invdepth_u16(pred_u16).astype(np.float64)
    gt_inv = decode_invdepth_u16(gt_u16).astype(np.float64)
    diff = pred_inv[mask] - gt_inv[mask]
    return {
        "proxy_inv_l1": float(np.abs(diff).mean()),
        "proxy_u16_mae": float(np.abs(pred[mask] - gt[mask]).mean()),
        "proxy_inv_rmse": float(np.sqrt(np.square(diff).mean())),
        "valid_pixels": int(mask.sum()),
    }


def summarize_scene(scene_rows):
    metric_keys = ["proxy_inv_l1", "proxy_u16_mae", "proxy_inv_rmse", "official_depth_epe", "official_psnr", "official_score"]
    summary = {
        key: float(np.mean([row[key] for row in scene_rows]))
        for key in metric_keys
    }
    official = [row["official_depth_epe"] for row in scene_rows]
    for local_key in ["proxy_inv_l1", "proxy_u16_mae", "proxy_inv_rmse"]:
        local = [row[local_key] for row in scene_rows]
        summary[f"{local_key}_pearson_vs_official_depth_epe"] = safe_pearson(local, official)
        summary[f"{local_key}_spearman_vs_official_depth_epe"] = safe_spearman(local, official)
    return summary


def evaluate_scene(scene, pred_root, proxy_gt_root, scene_root, log_info):
    expected = expected_test_depth_filenames(scene_root, scene)
    if sorted(expected) != sorted(log_info["filenames"]):
        raise RuntimeError(
            f"{scene} test view membership mismatch between images.bin and submission log.\n"
            f"Expected: {expected}\n"
            f"Log: {log_info['filenames']}"
        )

    filenames = log_info["filenames"]
    pred_depth_dir = resolve_prediction_depth_dir(pred_root, scene, filenames)
    gt_depth_dir = proxy_gt_root / scene / "depth"
    rows = []
    for filename, official in zip(filenames, log_info["metrics"]):
        pred_path = pred_depth_dir / filename
        gt_path = gt_depth_dir / filename
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing proxy GT file: {gt_path}")
        metrics = masked_metrics(load_u16_png(pred_path), load_u16_png(gt_path))
        row = {
            "scene": scene,
            "filename": filename,
            "prediction_dir": str(pred_depth_dir),
            "proxy_gt_path": str(gt_path),
            "official_depth_epe": official["depth_epe"],
            "official_psnr": official["psnr"],
            "official_score": official["score"],
        }
        row.update(metrics)
        rows.append(row)
    return pred_depth_dir, rows


def main():
    parser = ArgumentParser(description="Evaluate Track1 depth predictions against proxy GT and align them with submission log feedback.")
    parser.add_argument("--pred-root", type=Path, default=default_prediction_root())
    parser.add_argument("--proxy-gt-root", type=Path, default=default_proxy_gt_root())
    parser.add_argument("--scene-root", type=Path, default=default_scene_root())
    parser.add_argument("--log-path", type=Path, default=default_log_path())
    parser.add_argument("--label", type=str, default="3dgs_submit_track1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir(args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_entries = parse_submission_log(args.log_path)
    per_view = []
    summary = {"label": args.label, "pred_root": str(args.pred_root), "proxy_gt_root": str(args.proxy_gt_root), "log_path": str(args.log_path), "scenes": {}}

    scenes = list(args.scenes)
    for scene in scenes:
        pred_depth_dir, rows = evaluate_scene(scene, args.pred_root, args.proxy_gt_root, args.scene_root, log_entries[scene])
        per_view.extend(rows)
        summary["scenes"][scene] = summarize_scene(rows)
        summary["scenes"][scene]["prediction_dir"] = str(pred_depth_dir)

    overall_rows = per_view
    summary["overall"] = summarize_scene(overall_rows)

    json_path = output_dir / f"{args.label}_summary.json"
    per_view_json_path = output_dir / f"{args.label}_per_view.json"
    csv_path = output_dir / f"{args.label}_per_view.csv"

    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(per_view_json_path, "w") as handle:
        json.dump(per_view, handle, indent=2)
    with open(csv_path, "w", newline="") as handle:
        fieldnames = [
            "scene",
            "filename",
            "official_depth_epe",
            "official_psnr",
            "official_score",
            "proxy_inv_l1",
            "proxy_u16_mae",
            "proxy_inv_rmse",
            "valid_pixels",
            "prediction_dir",
            "proxy_gt_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_view)

    print(f"Wrote summary: {json_path}")
    print(f"Wrote per-view JSON: {per_view_json_path}")
    print(f"Wrote per-view CSV: {csv_path}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
