[简体中文](./README.md) | **English**

# gaussian-splatting-submit

This repository contains the I2WM project code for the NTIRE 2026 3D Content Super-Resolution Challenge.
It currently supports three main workflows:

- training from scratch or resuming from checkpoints
- rendering and exporting final competition-format `submit` results
- packaging the final Track 1 / Track 2 submissions into single-file bundles and reproducing them with bundle-only rendering

For the Chinese version, see:

- `README.md`

## 1. Current Capabilities

This submission-oriented codebase currently retains the following core functionality:

- the original training pipeline via `train.py`
- competition-mode training and rendering with fixed test views
- `points3D.txt -> cache points3D.ply` preprocessing
- phase-2 fine-tuning via `competition_phase2_iters`
- depth scale / bias affine adjustment inside `gaussian_renderer.render()`
- RGB / depth submission export
- Track 1 proxy-depth evaluation
- depth analysis and visualization tools
- Track 1 / Track 2 render-bundle build and replay

## 2. Environment

This project still relies on a CUDA / PyTorch environment similar to the original Gaussian Splatting setup.

At minimum, you should have:

- a PyTorch CUDA environment
- `diff_gaussian_rasterization`
- `simple_knn`

The environment specification is provided in:

- `environment.yml`

## 3. Dataset Layout

For training from scratch, each scene is expected to follow this structure:

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
- if you use the competition runner and your data is not stored under the default root, you should pass `--source-path-override`

## 4. Quick Start

### 4.1 Recommended Entry Point: Competition Runner

The recommended entry point is:

- `scripts/run_competition_scene.py`

Its default workflow includes:

- preparing the points3D cache
- training
- rendering `submit`
- computing RGB metrics

Minimal example:

```bash
python scripts/run_competition_scene.py \
  --track track1 \
  --scene EastResearchAreas
```

If your data is not located under the repository's default competition root, explicitly provide the source path:

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

Useful optional flags:

- `--prepare-only / --train-only / --render-only / --eval-only`
- `--run_render / --run_proxy_eval / --run_rgb_metrics`
- `--start-checkpoint`

### 4.2 Direct Training Entry

If you do not want to use the runner, you can still call:

- `train.py`

Minimal example:

```bash
python train.py \
  -s /path/to/scene \
  -m /path/to/output \
  --eval \
  --competition_mode \
  --points3d_cache_dir /path/to/points3d_cache \
  -d depth
```

This confirms that the repository still supports training from scratch; the bundle workflow is render-only and does not replace the training pipeline.

## 5. What `competition_mode` Means

`--competition_mode` is a boolean flag:

- omit it: default is `False`
- add it: it becomes `True`

When enabled, the code switches from the standard 3DGS evaluation logic to the competition workflow. In practice, it controls:

- fixed 10-view test-view handling
- competition points-cache logic
- competition-format inverse-depth PNG loading
- phase-2 fine-tuning on fixed test views only
- direct `submit` export during rendering

## 6. Rendering and Submission Export

After training, you can render using:

- `render.py`

or continue with:

- `scripts/run_competition_scene.py --render-only`

Under competition mode, the repository exports:

- `submit/`
- `submit_gt/`

## 7. Bundle-Based Reproduction

The repository can package the final Gaussian state, test-view camera metadata, and depth affine parameters into a single bundle file and reproduce the final outputs without relying on external `point_cloud.ply`, affine JSON files, `cameras.json`, or `source_path`.

If you only want to reproduce our final submissions, you can directly download the prepared bundles:

- [Track 1](https://drive.google.com/file/d/10LKLvezT7kgU0BiBiLvyuijypuazvcRZ/view?usp=sharing)
- [Track 2](https://drive.google.com/drive/folders/1VZmGpBKutgixbJjHm4KHs1IZh8GH4Jqk?usp=sharing)

After downloading, you can reproduce the outputs directly with:

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

If you need to rebuild the bundles locally:

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

Bundle rendering always writes:

- `<output-dir>/EastResearchAreas/submit`
- `<output-dir>/NorthAreas/submit`

Current bundle conventions:

- `track1`: `per-image + affine + l1 + png`
- `track2`: `scene + affine + l2 + raw`
- both tracks only package the final 10 test views
- bundles are render-only and do not support training resume

## 8. Additional Tools

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

## 9. Project Structure

Main files and directories:

- `train.py`: training entry point
- `render.py`: rendering entry point
- `scripts/run_competition_scene.py`: competition runner
- `scripts/build_track_bundle.py`: bundle build entry point
- `scripts/render_track_bundle.py`: bundle replay entry point
- `gaussian_renderer/`: renderer core
- `scene/`: scene and camera loading
- `competition_utils.py`: competition data path and points-cache utilities
- `competition_depth_utils.py`: depth affine and PNG encode/decode utilities

## 10. Notes

- This repository still retains training functionality; it is not just a submission shell
- The bundle workflow is a render-only path for reproduction and submission
- For more detailed cleanup, integration, and verification records, see `NTIRE2026_3DSR_WORKLOG.md`
