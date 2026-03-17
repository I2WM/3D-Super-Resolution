#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition_utils import (
    canonicalize_track,
    default_competition_model_path,
    default_points3d_cache_dir,
    ensure_competition_points3d_cache,
    resolve_source_path,
)


def tuning_root(track: str, campaign: str) -> Path:
    return REPO_ROOT / "workdirs" / "competition" / "tuning" / canonicalize_track(track) / campaign


def determine_actions(args) -> tuple[bool, bool, bool, bool, bool]:
    legacy_flags = [args.prepare_only, args.train_only, args.render_only, args.eval_only]
    if sum(bool(flag) for flag in legacy_flags) > 1:
        raise ValueError("Only one of --prepare-only, --train-only, --render-only, --eval-only may be set.")

    if any(legacy_flags):
        run_prepare = args.prepare_only or args.train_only
        run_train = args.train_only
        run_render = args.render_only
        run_proxy_eval = False
        run_rgb_metrics = False

        if args.train_only:
            run_render = bool(args.run_render)
            run_proxy_eval = bool(args.run_proxy_eval)
            run_rgb_metrics = bool(args.run_rgb_metrics)
        elif args.eval_only:
            run_proxy_eval = bool(args.run_proxy_eval)
            run_rgb_metrics = bool(args.run_rgb_metrics) or not args.run_proxy_eval

        if run_train and (run_proxy_eval or run_rgb_metrics):
            run_render = True
        return run_prepare, run_train, run_render, run_proxy_eval, run_rgb_metrics

    if any([args.run_render, args.run_proxy_eval, args.run_rgb_metrics]):
        run_prepare = True
        run_train = True
        run_render = bool(args.run_render)
        run_proxy_eval = bool(args.run_proxy_eval)
        run_rgb_metrics = bool(args.run_rgb_metrics)
        if run_proxy_eval or run_rgb_metrics:
            run_render = True
        return run_prepare, run_train, run_render, run_proxy_eval, run_rgb_metrics

    return True, True, True, False, True


def resolve_model_layout(track: str, scene: str, campaign: str | None, label: str | None) -> tuple[Path, Path, str]:
    if (campaign is None) != (label is None):
        raise ValueError("--campaign and --label must be provided together.")

    if campaign is None:
        model_path = default_competition_model_path(track, scene)
        logs_dir = model_path / "runner_logs"
        log_prefix = scene
        return model_path, logs_dir, log_prefix

    campaign_root = tuning_root(track, campaign)
    model_path = campaign_root / label / scene
    logs_dir = campaign_root / "logs"
    return model_path, logs_dir, label


def run_logged_command(cmd: list[str], cwd: Path, log_path: Path, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_str = " ".join(str(arg) for arg in cmd)
    print(f"Running: {command_str}")
    with log_path.open("w", encoding="utf-8") as handle:
        selected_gpu = env.get("CUDA_VISIBLE_DEVICES", "")
        handle.write(f"# physical_gpu={selected_gpu or 'unknown'}\n")
        handle.write(f"$ {command_str}\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def main() -> None:
    parser = ArgumentParser(description="Competition wrapper for NTIRE 2026 3DSR scenes")
    parser.add_argument("--track", required=True, type=str)
    parser.add_argument("--scene", required=True, type=str)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--competition_phase2_iters", type=int, default=10_000)
    parser.add_argument("--campaign", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--start-checkpoint", type=Path, default=None)
    parser.add_argument("--source-path-override", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--run_render", action="store_true")
    parser.add_argument("--run_proxy_eval", action="store_true")
    parser.add_argument("--run_rgb_metrics", action="store_true")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--no_depth", action="store_true")
    parser.add_argument("--depth_l1_weight_init", type=float, default=0.1)
    parser.add_argument("--depth_l1_weight_final", type=float, default=0.1)
    parser.add_argument("--disable_phase2_densify", action="store_true")
    parser.add_argument(
        "--competition_depth_adjustment",
        type=Path,
        default=None,
        help="Optional scale/bias json (or directory containing it) applied inside gaussian_renderer.render().",
    )
    args = parser.parse_args()

    track = canonicalize_track(args.track)
    scene = args.scene
    canonical_source_path = resolve_source_path(track, scene)
    source_path = args.source_path_override.resolve() if args.source_path_override is not None else canonical_source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source path: {source_path}")

    if args.start_checkpoint is not None and not args.start_checkpoint.exists():
        raise FileNotFoundError(f"Missing start checkpoint: {args.start_checkpoint}")

    run_prepare, run_train, run_render, run_proxy_eval, run_rgb_metrics = determine_actions(args)
    model_path, logs_dir, log_prefix = resolve_model_layout(track, scene, args.campaign, args.label)
    if args.campaign is not None and run_train and model_path.exists() and any(model_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing tuning output: {model_path}. "
            "Use a new --label to keep experiments isolated."
        )
    if run_proxy_eval and track != "track1":
        raise ValueError("--run_proxy_eval currently supports only track1.")

    points3d_cache_dir = default_points3d_cache_dir()
    final_iteration = args.iterations + args.competition_phase2_iters
    extra_env = {}
    if args.source_path_override is not None:
        extra_env["COMPETITION_SOURCE_PATH_FOR_CACHE"] = str(canonical_source_path)

    print(f"Track: {track}")
    print(f"Scene: {scene}")
    print(f"Canonical source path: {canonical_source_path}")
    print(f"Training source path: {source_path}")
    print(f"Model path: {model_path}")

    if run_prepare:
        cache_path, point_count, point_source = ensure_competition_points3d_cache(
            str(canonical_source_path),
            str(points3d_cache_dir),
        )
        print(f"Prepared points cache: {cache_path} ({point_count} points, source={point_source})")

    if run_train:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable,
            "train.py",
            "-s",
            str(source_path),
            "-m",
            str(model_path),
            "-r",
            str(args.resolution),
            "--eval",
            "--competition_mode",
            "--points3d_cache_dir",
            str(points3d_cache_dir),
            "--iterations",
            str(args.iterations),
            "--competition_phase2_iters",
            str(args.competition_phase2_iters),
            "--test_iterations",
            "7000",
            str(args.iterations),
            str(final_iteration),
            "--save_iterations",
            str(args.iterations),
            str(final_iteration),
            "--checkpoint_iterations",
            str(args.iterations),
            str(final_iteration),
            "--disable_viewer",
        ]
        if not args.no_depth:
            train_cmd.extend(
                [
                    "-d",
                    "depth",
                    "--depth_l1_weight_init",
                    str(args.depth_l1_weight_init),
                    "--depth_l1_weight_final",
                    str(args.depth_l1_weight_final),
                ]
            )
        if args.start_checkpoint is not None:
            train_cmd.extend(["--start_checkpoint", str(args.start_checkpoint)])
        if args.disable_phase2_densify:
            train_cmd.append("--disable_phase2_densify")
        run_logged_command(train_cmd, REPO_ROOT, logs_dir / f"{log_prefix}_train.log", extra_env=extra_env)

    if run_render:
        render_cmd = [
            sys.executable,
            "render.py",
            "-s",
            str(source_path),
            "-m",
            str(model_path),
            "--iteration",
            str(final_iteration),
            "--skip_train",
            "--competition_mode",
            "--points3d_cache_dir",
            str(points3d_cache_dir),
        ]
        if args.competition_depth_adjustment is not None:
            render_cmd.extend(
                [
                    "--competition_depth_adjustment",
                    str(args.competition_depth_adjustment),
                ]
            )
        run_logged_command(render_cmd, REPO_ROOT, logs_dir / f"{log_prefix}_render.log", extra_env=extra_env)

    if run_proxy_eval:
        proxy_eval_dir = model_path / "proxy_eval"
        proxy_eval_cmd = [
            sys.executable,
            "scripts/eval_track1_proxy_depth.py",
            "--pred-root",
            str(model_path.parent),
            "--scenes",
            scene,
            "--label",
            log_prefix,
            "--output-dir",
            str(proxy_eval_dir),
        ]
        run_logged_command(
            proxy_eval_cmd,
            REPO_ROOT,
            logs_dir / f"{log_prefix}_proxy_eval.log",
            extra_env=extra_env,
        )

    if run_rgb_metrics:
        metrics_cmd = [
            sys.executable,
            "metrics.py",
            "-m",
            str(model_path),
        ]
        run_logged_command(
            metrics_cmd,
            REPO_ROOT,
            logs_dir / f"{log_prefix}_rgb_metrics.log",
            extra_env=extra_env,
        )


if __name__ == "__main__":
    main()
