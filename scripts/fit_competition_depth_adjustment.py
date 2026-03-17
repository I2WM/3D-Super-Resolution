#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_depth_utils import (
    UINT16_MAX_INVDEPTH,
    apply_invdepth_affine,
    clip_png_invdepth,
    encode_invdepth_u16,
)
from depth_tooling_common import render_test_invdepths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit competition depth scale/affine adjustments for raw float invdepth before uint16 export."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-bias", action="store_true")
    parser.add_argument(
        "--fit-granularity",
        choices=("scene", "image"),
        default="scene",
        help="Whether to fit one parameter set for the full scene or one set per test image.",
    )
    parser.add_argument(
        "--fit-loss",
        choices=("l2", "l1"),
        default="l2",
        help="Loss used during fitting.",
    )
    parser.add_argument(
        "--fit-space",
        choices=("raw", "png"),
        default="raw",
        help="Whether to optimize raw float invdepth or PNG-equivalent clipped invdepth.",
    )
    parser.add_argument(
        "--nonnegative-scale",
        action="store_true",
        help="Constrain scale_a to be >= 0.0. Closed-form support is preserved for the old scene/raw/L2 path.",
    )
    parser.add_argument("--write-scaled-depths", action="store_true")
    return parser.parse_args()

def apply_adjustment(values: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return apply_invdepth_affine(values, scale=scale, bias=bias)


def high_clip_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        raise RuntimeError("Cannot compute clip ratio on empty values.")
    return float((values > UINT16_MAX_INVDEPTH).mean())


def low_clip_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        raise RuntimeError("Cannot compute clip ratio on empty values.")
    return float((values < PNG_MIN_INVDEPTH).mean())


def proxy_metrics(pred_values: np.ndarray, gt_values: np.ndarray) -> tuple[float, float]:
    diff = pred_values - gt_values
    return float(np.abs(diff).mean()), float(np.sqrt(np.square(diff).mean()))


def view_valid_records(
    pred_map: dict[str, np.ndarray], gt_map: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for name in sorted(pred_map):
        pred = pred_map[name]
        gt = gt_map[name]
        mask = gt > 0
        valid_pixels = int(mask.sum())
        if valid_pixels == 0:
            continue
        records.append(
            {
                "name": name,
                "pred_map": pred,
                "gt_map": gt,
                "pred_values": pred[mask].astype(np.float64, copy=False),
                "gt_values": gt[mask].astype(np.float64, copy=False),
                "valid_pixels": valid_pixels,
            }
        )
    if not records:
        raise RuntimeError("No valid depth pixels found across scene test views.")
    return records


def aggregate_valid_pixels(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, int]:
    pred_values = [record["pred_values"] for record in records]
    gt_values = [record["gt_values"] for record in records]
    valid_pixels = int(sum(int(record["valid_pixels"]) for record in records))
    return np.concatenate(pred_values), np.concatenate(gt_values), valid_pixels


def fit_scale_l2_raw(pred_values: np.ndarray, gt_values: np.ndarray) -> float:
    numerator = float(np.dot(pred_values, gt_values))
    denominator = float(np.dot(pred_values, pred_values))
    if denominator <= 0:
        raise RuntimeError("Degenerate scale fit: sum(pred * pred) is not positive.")
    scale = numerator / denominator
    if not np.isfinite(scale):
        raise RuntimeError(f"Degenerate scale fit: non-finite scale {scale}")
    return float(scale)


def fit_affine_l2_raw(pred_values: np.ndarray, gt_values: np.ndarray) -> tuple[float, float]:
    design = np.stack(
        [pred_values.astype(np.float64, copy=False), np.ones_like(pred_values, dtype=np.float64)],
        axis=1,
    )
    target = gt_values.astype(np.float64, copy=False)
    solution, residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    if rank < 2:
        raise RuntimeError(
            "Degenerate affine fit: least squares system is rank deficient. "
            f"rank={rank}, singular_values={singular_values.tolist()}"
        )
    scale = float(solution[0])
    bias = float(solution[1])
    if not np.isfinite(scale) or not np.isfinite(bias):
        raise RuntimeError(
            f"Degenerate affine fit: non-finite parameters scale={scale}, bias={bias}"
        )
    return scale, bias


def fit_affine_nonnegative_scale_l2_raw(
    pred_values: np.ndarray, gt_values: np.ndarray
) -> tuple[float, float, bool]:
    scale, bias = fit_affine_l2_raw(pred_values, gt_values)
    if scale >= 0.0:
        return scale, bias, False

    boundary_bias = float(gt_values.astype(np.float64, copy=False).mean())
    if not np.isfinite(boundary_bias):
        raise RuntimeError(
            f"Degenerate constrained affine fit: non-finite boundary bias {boundary_bias}"
        )
    return 0.0, boundary_bias, True


def metric_space_values(values: np.ndarray, fit_space: str) -> np.ndarray:
    if fit_space == "png":
        return clip_png_invdepth(values)
    return values


def optimization_loss(
    adjusted_values: np.ndarray, gt_values: np.ndarray, fit_loss: str, fit_space: str
) -> float:
    compared = metric_space_values(adjusted_values, fit_space)
    diff = compared - gt_values
    if fit_loss == "l1":
        return float(np.abs(diff).mean())
    return float(np.square(diff).mean())


def unique_starts(starts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for scale, bias in starts:
        key = (round(float(scale), 12), round(float(bias), 12))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((float(scale), float(bias)))
    return deduped


def build_multistarts(
    pred_values: np.ndarray, gt_values: np.ndarray, fit_bias: bool
) -> list[tuple[float, float]]:
    starts: list[tuple[float, float]] = [(1.0, 0.0), (0.0, float(gt_values.mean()))]
    scale_only = fit_scale_l2_raw(pred_values, gt_values)
    starts.append((scale_only, 0.0))
    if fit_bias:
        affine_scale, affine_bias = fit_affine_l2_raw(pred_values, gt_values)
        starts.append((affine_scale, affine_bias))
    return unique_starts(starts)


def optimize_adjustment(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    fit_bias: bool,
    fit_loss: str,
    fit_space: str,
    nonnegative_scale: bool,
) -> tuple[float, float, bool, dict[str, object]]:
    starts = build_multistarts(pred_values, gt_values, fit_bias)
    best_solution: tuple[float, float] | None = None
    best_objective = float("inf")
    best_constraint_activated = False
    start_summaries = []

    for start_scale, start_bias in starts:
        x0 = np.array([start_scale, start_bias] if fit_bias else [start_scale], dtype=np.float64)

        def objective(x: np.ndarray) -> float:
            scale = float(x[0])
            bias = float(x[1]) if fit_bias else 0.0
            if nonnegative_scale and scale < 0.0:
                return float("inf")
            adjusted = apply_adjustment(pred_values, scale, bias)
            return optimization_loss(adjusted, gt_values, fit_loss, fit_space)

        result = optimize.minimize(
            objective,
            x0=x0,
            method="Powell",
            options={"xtol": 1e-6, "ftol": 1e-9, "maxiter": 200, "maxfev": 2000},
        )

        scale = float(result.x[0])
        bias = float(result.x[1]) if fit_bias else 0.0
        constraint_activated = False
        if nonnegative_scale and scale < 0.0:
            scale = 0.0
            constraint_activated = True

        objective_value = objective(np.array([scale, bias] if fit_bias else [scale], dtype=np.float64))
        start_summaries.append(
            {
                "start_scale_a": float(start_scale),
                "start_bias_b": float(start_bias if fit_bias else 0.0),
                "final_scale_a": scale,
                "final_bias_b": bias,
                "objective": objective_value,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nit": getattr(result, "nit", None),
                "nfev": getattr(result, "nfev", None),
                "constraint_activated": constraint_activated,
            }
        )

        if objective_value < best_objective:
            best_objective = objective_value
            best_solution = (scale, bias)
            best_constraint_activated = constraint_activated

    if best_solution is None:
        raise RuntimeError("Optimization failed to produce any candidate solution.")

    return best_solution[0], best_solution[1], best_constraint_activated, {
        "starts": start_summaries,
        "best_objective": best_objective,
    }


def fit_one_unit(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    fit_bias: bool,
    fit_loss: str,
    fit_space: str,
    nonnegative_scale: bool,
) -> tuple[float, float, bool, dict[str, object]]:
    if fit_loss == "l2" and fit_space == "raw":
        if fit_bias:
            if nonnegative_scale:
                scale, bias, constraint_activated = fit_affine_nonnegative_scale_l2_raw(
                    pred_values, gt_values
                )
                return scale, bias, constraint_activated, {
                    "solver": "closed_form_l2_raw_affine_nonnegative"
                }
            scale, bias = fit_affine_l2_raw(pred_values, gt_values)
            return scale, bias, False, {"solver": "closed_form_l2_raw_affine"}
        scale = fit_scale_l2_raw(pred_values, gt_values)
        return scale, 0.0, False, {"solver": "closed_form_l2_raw_scale_only"}

    scale, bias, constraint_activated, optimization_info = optimize_adjustment(
        pred_values=pred_values,
        gt_values=gt_values,
        fit_bias=fit_bias,
        fit_loss=fit_loss,
        fit_space=fit_space,
        nonnegative_scale=nonnegative_scale,
    )
    return scale, bias, constraint_activated, {"solver": "powell", **optimization_info}


def evaluate_one_unit(
    name: str,
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    scale: float,
    bias: float,
    fit_loss: str,
    fit_space: str,
    valid_pixels: int,
    constraint_activated: bool,
    solver_info: dict[str, object],
) -> dict[str, object]:
    adjusted_unclipped = apply_adjustment(pred_values, scale, bias)
    png_equivalent_raw = clip_png_invdepth(pred_values)
    adjusted_png = clip_png_invdepth(adjusted_unclipped)

    raw_l1, raw_rmse = proxy_metrics(pred_values, gt_values)
    png_l1, png_rmse = proxy_metrics(png_equivalent_raw, gt_values)
    scaled_l1, scaled_rmse = proxy_metrics(adjusted_png, gt_values)

    objective = optimization_loss(adjusted_unclipped, gt_values, fit_loss, fit_space)
    warning_scale_not_shrinking = bool(scale >= 1.0)

    return {
        "file": name,
        "scale_a": float(scale),
        "bias_b": float(bias),
        "valid_pixels": int(valid_pixels),
        "constraint_activated": bool(constraint_activated),
        "warning_scale_not_shrinking": warning_scale_not_shrinking,
        "fit_objective": float(objective),
        "raw_proxy_inv_l1": raw_l1,
        "raw_proxy_inv_rmse": raw_rmse,
        "png_equivalent_proxy_inv_l1": png_l1,
        "png_equivalent_proxy_inv_rmse": png_rmse,
        "scaled_proxy_inv_l1": scaled_l1,
        "scaled_proxy_inv_rmse": scaled_rmse,
        "raw_clip_ratio": high_clip_ratio(pred_values),
        "scaled_clip_ratio": high_clip_ratio(adjusted_unclipped),
        "raw_low_clip_ratio": low_clip_ratio(pred_values),
        "scaled_low_clip_ratio": low_clip_ratio(adjusted_unclipped),
        "solver_info": solver_info,
    }


def fit_records(records: list[dict[str, object]], args: argparse.Namespace) -> list[dict[str, object]]:
    if args.fit_granularity == "scene":
        pred_values, gt_values, _ = aggregate_valid_pixels(records)
        scale, bias, constraint_activated, solver_info = fit_one_unit(
            pred_values=pred_values,
            gt_values=gt_values,
            fit_bias=args.fit_bias,
            fit_loss=args.fit_loss,
            fit_space=args.fit_space,
            nonnegative_scale=args.nonnegative_scale,
        )
        per_view_fits = []
        for record in records:
            per_view_fits.append(
                evaluate_one_unit(
                    name=str(record["name"]),
                    pred_values=record["pred_values"],
                    gt_values=record["gt_values"],
                    scale=scale,
                    bias=bias,
                    fit_loss=args.fit_loss,
                    fit_space=args.fit_space,
                    valid_pixels=int(record["valid_pixels"]),
                    constraint_activated=constraint_activated,
                    solver_info=solver_info,
                )
            )
        return per_view_fits

    per_view_fits = []
    for record in records:
        scale, bias, constraint_activated, solver_info = fit_one_unit(
            pred_values=record["pred_values"],
            gt_values=record["gt_values"],
            fit_bias=args.fit_bias,
            fit_loss=args.fit_loss,
            fit_space=args.fit_space,
            nonnegative_scale=args.nonnegative_scale,
        )
        per_view_fits.append(
            evaluate_one_unit(
                name=str(record["name"]),
                pred_values=record["pred_values"],
                gt_values=record["gt_values"],
                scale=scale,
                bias=bias,
                fit_loss=args.fit_loss,
                fit_space=args.fit_space,
                valid_pixels=int(record["valid_pixels"]),
                constraint_activated=constraint_activated,
                solver_info=solver_info,
            )
        )
    return per_view_fits


def per_view_fit_map(per_view_fits: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    return {
        str(entry["file"]): (float(entry["scale_a"]), float(entry["bias_b"]))
        for entry in per_view_fits
    }


def write_scaled_depths(
    pred_map: dict[str, np.ndarray], fit_map: dict[str, tuple[float, float]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in pred_map.items():
        scale, bias = fit_map[name]
        scaled = clip_png_invdepth(apply_adjustment(pred.astype(np.float64, copy=False), scale, bias))
        scaled_u16 = encode_invdepth_u16(scaled)
        stem = Path(name).stem
        Image.fromarray(scaled_u16).save(output_dir / f"{stem}_depth.png")


def aggregate_metrics(
    records: list[dict[str, object]], fit_map: dict[str, tuple[float, float]]
) -> dict[str, float]:
    raw_values = []
    gt_values = []
    scaled_unclipped_values = []
    for record in records:
        pred = record["pred_values"]
        gt = record["gt_values"]
        scale, bias = fit_map[str(record["name"])]
        raw_values.append(pred)
        gt_values.append(gt)
        scaled_unclipped_values.append(apply_adjustment(pred, scale, bias))

    raw = np.concatenate(raw_values)
    gt = np.concatenate(gt_values)
    scaled_unclipped = np.concatenate(scaled_unclipped_values)
    png_equivalent_raw = clip_png_invdepth(raw)
    scaled_png = clip_png_invdepth(scaled_unclipped)

    raw_l1, raw_rmse = proxy_metrics(raw, gt)
    png_l1, png_rmse = proxy_metrics(png_equivalent_raw, gt)
    scaled_l1, scaled_rmse = proxy_metrics(scaled_png, gt)

    return {
        "valid_pixels": int(raw.size),
        "raw_clip_ratio": high_clip_ratio(raw),
        "png_equivalent_clip_ratio": high_clip_ratio(raw),
        "scaled_clip_ratio": high_clip_ratio(scaled_unclipped),
        "raw_low_clip_ratio": low_clip_ratio(raw),
        "scaled_low_clip_ratio": low_clip_ratio(scaled_unclipped),
        "raw_proxy_inv_l1": raw_l1,
        "raw_proxy_inv_rmse": raw_rmse,
        "png_equivalent_proxy_inv_l1": png_l1,
        "png_equivalent_proxy_inv_rmse": png_rmse,
        "scaled_proxy_inv_l1": scaled_l1,
        "scaled_proxy_inv_rmse": scaled_rmse,
        "warning_scale_not_shrinking": bool(
            any(scale >= 1.0 for scale, _ in fit_map.values())
        ),
        "num_scales_not_shrinking": int(sum(scale >= 1.0 for scale, _ in fit_map.values())),
    }


def write_per_view_csv(per_view_fits: list[dict[str, object]], output_path: Path) -> None:
    fields = [
        "file",
        "scale_a",
        "bias_b",
        "valid_pixels",
        "constraint_activated",
        "warning_scale_not_shrinking",
        "fit_objective",
        "raw_proxy_inv_l1",
        "raw_proxy_inv_rmse",
        "png_equivalent_proxy_inv_l1",
        "png_equivalent_proxy_inv_rmse",
        "scaled_proxy_inv_l1",
        "scaled_proxy_inv_rmse",
        "raw_clip_ratio",
        "scaled_clip_ratio",
        "raw_low_clip_ratio",
        "scaled_low_clip_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in per_view_fits:
            writer.writerow({field: entry[field] for field in fields})


def main() -> None:
    args = parse_args()
    if args.nonnegative_scale and not args.fit_bias:
        raise RuntimeError("--nonnegative-scale requires --fit-bias.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_map, gt_map, cfg = render_test_invdepths(args.model_path, args.iteration)
    source_scene = Path(cfg.source_path).name
    if source_scene != args.scene:
        raise RuntimeError(
            f"Scene mismatch: --scene={args.scene}, but cfg_args source_path points to {source_scene}"
        )

    records = view_valid_records(pred_map, gt_map)
    per_view_fits = fit_records(records, args)
    fit_map = per_view_fit_map(per_view_fits)
    aggregate = aggregate_metrics(records, fit_map)

    scaled_submit_dir = args.output_dir / "scaled_submit"
    if args.write_scaled_depths:
        write_scaled_depths(pred_map, fit_map, scaled_submit_dir)

    fit_mode = "affine" if args.fit_bias else "scale_only"
    constraint_activated = bool(any(bool(entry["constraint_activated"]) for entry in per_view_fits))

    if args.fit_granularity == "scene":
        top_level_scale = float(per_view_fits[0]["scale_a"])
        top_level_bias = float(per_view_fits[0]["bias_b"])
    else:
        top_level_scale = None
        top_level_bias = None

    result = {
        "scene": args.scene,
        "iteration": args.iteration,
        "model_path": str(args.model_path.resolve()),
        "source_path": cfg.source_path,
        "fit_mode": fit_mode,
        "fit_granularity": args.fit_granularity,
        "fit_loss": args.fit_loss,
        "fit_space": args.fit_space,
        "scale_a": top_level_scale,
        "bias_b": top_level_bias,
        "nonnegative_scale_constraint": bool(args.nonnegative_scale),
        "constraint_activated": constraint_activated,
        "valid_pixels": aggregate["valid_pixels"],
        "clip_threshold": float(UINT16_MAX_INVDEPTH),
        "raw_clip_ratio": aggregate["raw_clip_ratio"],
        "scaled_clip_ratio": aggregate["scaled_clip_ratio"],
        "raw_low_clip_ratio": aggregate["raw_low_clip_ratio"],
        "scaled_low_clip_ratio": aggregate["scaled_low_clip_ratio"],
        "raw_proxy_inv_l1": aggregate["raw_proxy_inv_l1"],
        "png_equivalent_proxy_inv_l1": aggregate["png_equivalent_proxy_inv_l1"],
        "scaled_proxy_inv_l1": aggregate["scaled_proxy_inv_l1"],
        "raw_proxy_inv_rmse": aggregate["raw_proxy_inv_rmse"],
        "png_equivalent_proxy_inv_rmse": aggregate["png_equivalent_proxy_inv_rmse"],
        "scaled_proxy_inv_rmse": aggregate["scaled_proxy_inv_rmse"],
        "num_test_views": len(records),
        "warning_scale_not_shrinking": aggregate["warning_scale_not_shrinking"],
        "num_scales_not_shrinking": aggregate["num_scales_not_shrinking"],
        "scaled_submit_dir": str(scaled_submit_dir.resolve()) if args.write_scaled_depths else None,
        "per_view_fits": per_view_fits,
    }

    result_text = json.dumps(result, indent=2)
    per_view_text = json.dumps(per_view_fits, indent=2)
    (args.output_dir / "depth_adjustment.json").write_text(result_text)
    (args.output_dir / "scale_fit.json").write_text(result_text)
    (args.output_dir / "per_view_depth_adjustments.json").write_text(per_view_text)
    (args.output_dir / "per_view_fits.json").write_text(per_view_text)
    write_per_view_csv(per_view_fits, args.output_dir / "per_view_fits.csv")

    if aggregate["warning_scale_not_shrinking"]:
        print(
            f"[WARN] {aggregate['num_scales_not_shrinking']} fitted scale(s) are >= 1.0. "
            "This does not support the 'shrink before uint16 export' hypothesis."
        )
    if constraint_activated:
        print(
            "[INFO] At least one fit preferred a negative scale and was clamped by the "
            "nonnegative-scale constraint."
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
