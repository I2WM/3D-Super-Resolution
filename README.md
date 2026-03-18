# gaussian-splatting-submit

Our solution consists of a **2D super-resolution front-end** and a **3DGS reconstruction back-end**.
For the front-end models used in both tracks, please refer to [Trident](https://github.com/I2WM/Trident) for model code and pretrained checkpoints.
This repository mainly contains the 3DGS reconstruction back-end, including the training pipeline, competition-format rendering/export, and bundle-based reproduction of the final submissions.

## Overview

This repository supports three main workflows:

- training from scratch or resuming from checkpoints
- rendering and exporting competition-format `submit` / `submit_gt` outputs
- packaging the final submissions into single-file bundles and reproducing them through bundle-only rendering

## Feature Summary

The current codebase retains and validates the following functionality:

- the original `train.py` training pipeline
- competition-mode training and rendering
- fixed test-view splitting and phase-2 fine-tuning
- `points3D.txt -> cache points3D.ply` preprocessing
- render-time depth affine adjustment
- RGB / depth submission export
- Track 1 proxy-depth evaluation
- depth analysis and visualization utilities
- Track 1 / Track 2 bundle build and bundle-only replay

## Entry Points

| Entry point | Purpose |
| --- | --- |
| `train.py` | Direct training or resume from checkpoints |
| `render.py` | Standard rendering and competition-format export |
| `scripts/run_competition_scene.py` | Recommended competition workflow wrapper for prepare/train/render/eval |
| `scripts/build_track_bundle.py` | Pack final results into a single-file bundle |
| `scripts/render_track_bundle.py` | Reproduce final outputs directly from a bundle |

## Environment

The project depends on a CUDA / PyTorch setup similar to the original Gaussian Splatting environment.

At minimum, you should have:

- a PyTorch CUDA environment
- `diff_gaussian_rasterization`
- `simple_knn`

The environment specification is provided in:

- `environment.yml`

## Dataset Layout

For training from scratch, each scene is expected to follow this layout:

```text
<scene>/
  images/
  depth/
  sparse/0/
    images.bin or images.txt
    cameras.bin or cameras.txt
    points3D.txt or points3D.ply
```

When `competition_mode` is enabled:

- depth is read from competition-format `*_depth.png`
- initialization points are preferably generated from `points3D.txt`
- if you use `scripts/run_competition_scene.py` and your data is not under the default competition root, pass `--source-path-override`

## Quick Start

### Recommended Workflow: Competition Runner

The recommended entry point is `scripts/run_competition_scene.py`.
By default, it can chain together:

- points-cache preparation
- training
- `submit` rendering
- RGB metrics computation

Minimal example:

```bash
python scripts/run_competition_scene.py \
  --track track1 \
  --scene EastResearchAreas
```

If your data is stored outside the default competition root:

```bash
python scripts/run_competition_scene.py \
  --track track2 \
  --scene NorthAreas \
  --source-path-override /path/to/scene
```

Example with phase-2 fine-tuning, depth loss, and render-time depth adjustment:

```bash
python scripts/run_competition_scene.py \
  --track track1 \
  --scene EastResearchAreas \
  --iterations 30000 \
  --competition_phase2_iters 10000 \
  --depth_l1_weight_init 0.1 \
  --depth_l1_weight_final 0.1 \
  --competition_depth_adjustment /path/to/depth_adjustment.json
```

Useful control flags:

- `--prepare-only`
- `--train-only`
- `--render-only`
- `--eval-only`
- `--run_render`
- `--run_proxy_eval`
- `--run_rgb_metrics`
- `--start-checkpoint`

### Direct Training

If you do not want to use the wrapper, you can still call `train.py` directly:

```bash
python train.py \
  -s /path/to/scene \
  -m /path/to/output \
  --eval \
  --competition_mode \
  --points3d_cache_dir /path/to/points3d_cache \
  -d depth
```

This path still supports full training. The bundle workflow is render-only and does not replace the training pipeline.

## What `competition_mode` Means

`--competition_mode` is a boolean flag:

- if omitted, it defaults to `False`
- if provided, it switches to `True`

When enabled, the repository switches to the competition workflow. In practice, it changes:

- fixed 10-view test-view handling
- competition-specific points-cache logic
- competition-format inverse-depth PNG loading
- phase-2 fine-tuning on fixed test views only
- direct `submit` export during rendering

## Rendering and Submission Export

After training, competition-format outputs can be generated in either of these ways:

- `python render.py ...`
- `python scripts/run_competition_scene.py --render-only ...`

Under `competition_mode`, the output layout contains:

- `submit/`
- `submit_gt/`

## Bundle-Based Reproduction

The repository can package the final Gaussian state, test-view camera metadata, and depth affine parameters into a single bundle file.
Bundle-only rendering does not depend on external:

- `point_cloud.ply`
- affine JSON files
- `cameras.json`
- `source_path`

If you only want to reproduce our final submissions, you can download the prepared bundle checkpoints directly:

- [Track 1](https://drive.google.com/file/d/10LKLvezT7kgU0BiBiLvyuijypuazvcRZ/view?usp=sharing)
- [Track 2](https://drive.google.com/drive/folders/1VZmGpBKutgixbJjHm4KHs1IZh8GH4Jqk?usp=sharing)

Render them directly with:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/render_track_bundle.py \
  --bundle /path/to/downloaded_track1_bundle.ckpt \
  --output-dir /path/to/track1_render
```

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/render_track_bundle.py \
  --bundle /path/to/downloaded_track2_bundle.ckpt \
  --output-dir /path/to/track2_render
```

To rebuild bundles locally:

```bash
python scripts/build_track_bundle.py \
  --track track1 \
  --output /tmp/track1_bundle.ckpt
```

```bash
python scripts/build_track_bundle.py \
  --track track2 \
  --output /tmp/track2_bundle.ckpt
```

Bundle rendering always writes to:

- `<output-dir>/EastResearchAreas/submit`
- `<output-dir>/NorthAreas/submit`

Current bundle conventions:

- `track1`: `per-image + affine + l1 + png`
- `track2`: `scene + affine + l2 + raw`
- both tracks package only the final 10 test views
- bundles are render-only artifacts and do not support training resume

## Additional Tools

Depth-adjustment fitting:

```bash
python scripts/fit_competition_depth_adjustment.py \
  --model-path workdirs/competition/exp/track1/EastResearchAreas \
  --iteration 40000 \
  --scene EastResearchAreas \
  --output-dir workdirs/competition/analysis/depth_adjustment/east
```

Track 1 proxy-depth evaluation:

```bash
python scripts/eval_track_proxy_depth.py \
  --pred-root workdirs/competition/exp/track1 \
  --label track1_baseline
```

Depth analysis and visualization:

- `scripts/compare_raw_invdepth_three.py`
- `scripts/visualize_submit_depth.py`
- `scripts/visualize_three_depth_dirs.py`

## Project Structure

```text
gaussian-splatting-submit/
├── train.py
├── render.py
├── gaussian_renderer/
├── scene/
├── competition_utils.py
├── competition_depth_utils.py
├── scripts/
│   ├── run_competition_scene.py
│   ├── build_track_bundle.py
│   ├── render_track_bundle.py
│   ├── fit_competition_depth_adjustment.py
│   ├── eval_track_proxy_depth.py
│   └── ...
├── README.md
└── NTIRE2026_3DSR_WORKLOG.md
```
