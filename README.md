**简体中文** | [English](./README_EN.md)

# gaussian-splatting-submit

这是 I2WM 在 NTIRE 2026 3D Content Super-Resolution Challenge 中的项目代码， 
它现在可以做三件核心事情：

- 从头训练或从 checkpoint 继续训练
- 按比赛口径渲染并导出 `submit`
- 把 I2WM 团队最终提交所需内容打成单文件 bundle，并只靠 bundle + 这份代码完成 render

英文版说明见：

- `README_EN.md`

## 1. 当前保留的能力

这个提交副本当前保留并支持以下核心能力：

- 原始训练主链：可以从头训练，也可以从已有 checkpoint 继续训练
- 比赛模式训练：支持固定 test views、points cache、phase2 fine-tune 和比赛口径的 depth 读取
- 提交渲染：支持按比赛格式导出 `submit` 和 `submit_gt`
- Bundle 复现：支持把最终提交所需内容打成单文件 bundle，并只靠 bundle + 代码完成 render

## 2. 环境依赖

项目仍然依赖与原始 Gaussian Splatting 类似的 CUDA / PyTorch 环境。

建议准备：

- PyTorch CUDA 环境
- `diff_gaussian_rasterization`
- `simple_knn`

环境文件见：

- `environment.yml`

## 3. 数据组织

从头训练时，单个 scene 至少需要满足以下结构：

```text
<scene>/
  images/
  depth/
  sparse/0/
    images.bin or images.txt
    cameras.bin or cameras.txt
    points3D.txt or points3D.ply
```

在 `competition_mode` 下：

- depth 使用比赛口径的 `*_depth.png`
- 初始化点云会优先从 `points3D.txt` 生成 cache
- 如果使用比赛 runner，数据不在默认根目录时应显式传入 `--source-path-override`

## 4. 快速开始

### 4.1 推荐入口：比赛 runner

推荐使用：

- `scripts/run_competition_scene.py`

默认流程包括：

- 准备 points3D cache
- 训练
- 渲染 submit 结果
- 计算 RGB metrics

最常用示例：

```bash
python scripts/run_competition_scene.py \
  --track track1 \
  --scene EastResearchAreas
```

如果数据不在代码默认的比赛根目录下，建议显式指定：

```bash
python scripts/run_competition_scene.py \
  --track track2 \
  --scene NorthAreas \
  --source-path-override /path/to/scene
```

带 phase2、depth loss 和 render 内 depth adjustment 的示例：

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

常用附加开关：

- `--prepare-only / --train-only / --render-only / --eval-only`
- `--run_render / --run_proxy_eval / --run_rgb_metrics`
- `--start-checkpoint`

### 4.2 直接使用训练入口

如果不想走 runner，也可以直接调用：

- `train.py`

最小示例：

```bash
python train.py \
  -s /path/to/scene \
  -m /path/to/output \
  --eval \
  --competition_mode \
  --points3d_cache_dir /path/to/points3d_cache \
  -d depth
```

这说明当前仓库仍然保留了从头训练的能力；bundle 脚本是 render-only 工具，不会替代训练主链。

## 5. `competition_mode` 说明

`--competition_mode` 是一个布尔开关：

- 不写它：默认 `False`
- 写上它：变成 `True`

打开后，代码会切到比赛流程，而不是普通 3DGS 评估流程。它主要控制：

- 固定 10 个 test views
- 比赛 points cache 逻辑
- 比赛口径的 invdepth PNG 读取
- phase2 只对固定 test views 再 fine-tune
- render 时直接导出 `submit`

## 6. 渲染与提交导出

训练完成后，可以直接使用：

- `render.py`

或者继续走：

- `scripts/run_competition_scene.py --render-only`

比赛模式下会生成：

- `submit/`
- `submit_gt/`

## 7. Bundle 复现

当前仓库支持把最终提交所需的高斯状态、test-view 相机和 depth affine 参数打成单文件 bundle，并在不依赖外部 `point_cloud.ply / affine json / cameras.json / source_path` 的情况下直接 render。

如果只想直接复现我们的最终提交结果，可以直接下载已经整理好的 bundle checkpoints：

- [Track 1](https://drive.google.com/file/d/10LKLvezT7kgU0BiBiLvyuijypuazvcRZ/view?usp=sharing)
- [Track 2](https://drive.google.com/drive/folders/1VZmGpBKutgixbJjHm4KHs1IZh8GH4Jqk?usp=sharing)

下载后，无需重新训练，也无需准备外部 `point_cloud.ply / affine json / cameras.json / source_path`，直接使用下面的命令即可复现：

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

如果需要重新构包，可以使用：

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

Bundle 渲染的输出目录统一为：

- `<output-dir>/EastResearchAreas/submit`
- `<output-dir>/NorthAreas/submit`

当前 bundle 约定：

- `track1`：`per-image + affine + l1 + png`
- `track2`：`scene + affine + l2 + raw`
- 两个 track 都只打包最终 10 个 test views
- bundle 是 render-only，不保留 resume training 能力

## 8. 其他工具

Depth adjustment 拟合：

```bash
python scripts/fit_competition_depth_adjustment.py \
  --model-path workdirs/competition/exp/track1/EastResearchAreas \
  --iteration 40000 \
  --scene EastResearchAreas \
  --output-dir workdirs/competition/analysis/depth_adjustment/east
```

Track1 proxy depth 评估：

```bash
python scripts/eval_track_proxy_depth.py \
  --pred-root workdirs/competition/exp/track1 \
  --label track1_baseline
```

深度分析与可视化：

- `scripts/compare_raw_invdepth_three.py`
- `scripts/visualize_submit_depth.py`
- `scripts/visualize_three_depth_dirs.py`

## 9. 项目结构

主要文件与目录如下：

- `train.py`：训练入口
- `render.py`：渲染入口
- `scripts/run_competition_scene.py`：比赛 runner
- `scripts/build_track_bundle.py`：构包入口
- `scripts/render_track_bundle.py`：bundle 渲染入口
- `gaussian_renderer/`：渲染核心
- `scene/`：场景与相机加载
- `competition_utils.py`：比赛数据路径与 points cache 工具
- `competition_depth_utils.py`：depth affine 与 PNG 编解码工具

## 10. 备注

- 这份仓库仍然保留训练能力，不是只有提交壳子
- bundle 流程是为最终复现和提交流程服务的 render-only 路径
- 更详细的清理、合并和验证记录见 `NTIRE2026_3DSR_WORKLOG.md`
