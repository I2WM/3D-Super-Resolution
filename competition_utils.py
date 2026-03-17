import os
from pathlib import Path
import shutil

import numpy as np


COMPETITION_DATA_ROOT = Path("/tmp/scratch-space/ntire2026/3dsr/test_trainset")
TESTVIEW_IDS = [6, 16, 22, 25, 39, 50, 51, 52, 66, 71]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def canonicalize_track(track: str) -> str:
    normalized = str(track).strip().lower()
    if normalized in {"1", "track1", "3dsr_track1"}:
        return "track1"
    if normalized in {"2", "track2", "3dsr_track2"}:
        return "track2"
    raise ValueError(f"Unsupported track '{track}'. Expected track1 or track2.")


def competition_track_dir(track: str) -> str:
    return f"3dsr_{canonicalize_track(track)}"


def resolve_source_path(track: str, scene: str) -> Path:
    source_path = COMPETITION_DATA_ROOT / competition_track_dir(track) / scene
    if not source_path.exists():
        raise FileNotFoundError(f"Competition scene not found: {source_path}")
    return source_path


def default_competition_model_path(track: str, scene: str) -> Path:
    return repo_root() / "workdirs" / "competition" / "exp" / canonicalize_track(track) / scene


def default_points3d_cache_dir() -> Path:
    return repo_root() / "workdirs" / "competition" / "cache" / "points3D"


def infer_track_and_scene(source_path: str):
    override_source_path = os.environ.get("COMPETITION_SOURCE_PATH_FOR_CACHE", "").strip()
    source = Path(override_source_path or source_path).resolve()
    scene = source.name
    track_dir = source.parent.name
    if track_dir == "3dsr_track1":
        return "track1", scene
    if track_dir == "3dsr_track2":
        return "track2", scene
    raise ValueError(f"Cannot infer competition track/scene from source path: {source}")


def cache_points3d_path(source_path: str, points3d_cache_dir: str) -> Path:
    track, scene = infer_track_and_scene(source_path)
    return Path(points3d_cache_dir) / track / scene / "points3D.ply"


def count_ply_vertices(ply_path: Path) -> int:
    with open(ply_path, "rb") as handle:
        first_line = handle.readline().decode("ascii", errors="strict").strip()
        if first_line != "ply":
            raise RuntimeError(f"Invalid PLY header in {ply_path}: missing 'ply' magic.")
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise RuntimeError(f"Failed to locate vertex count in PLY header: {ply_path}")


def read_points3d_text(path: Path):
    num_points = 0
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                elems = line.split()
                xyzs[count] = np.array(tuple(map(float, elems[1:4])))
                rgbs[count] = np.array(tuple(map(int, elems[4:7])))
                errors[count] = float(elems[7])
                count += 1
    return xyzs, rgbs, errors


def store_ply(path: Path, xyz, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz to have shape [N, 3], got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb to have shape [N, 3], got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb row count mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    with open(path, "w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float nx\n")
        handle.write("property float ny\n")
        handle.write("property float nz\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb, strict=True):
            handle.write(
                f"{float(point[0]):.9f} {float(point[1]):.9f} {float(point[2]):.9f} "
                f"0.0 0.0 0.0 {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def ensure_competition_points3d_cache(source_path: str, points3d_cache_dir: str):
    source = Path(source_path)
    sparse_dir = source / "sparse" / "0"
    txt_path = sparse_dir / "points3D.txt"
    external_ply_path = sparse_dir / "points3D.ply"
    cache_ply_path = cache_points3d_path(source_path, points3d_cache_dir)

    if txt_path.exists():
        needs_refresh = not cache_ply_path.exists() or cache_ply_path.stat().st_mtime < txt_path.stat().st_mtime
        if needs_refresh:
            print(f"Preparing competition points cache from {txt_path} -> {cache_ply_path}")
            xyz, rgb, _ = read_points3d_text(txt_path)
            store_ply(cache_ply_path, xyz, rgb)
            point_count = int(xyz.shape[0])
        else:
            point_count = count_ply_vertices(cache_ply_path)
        return cache_ply_path, point_count, "txt"

    if external_ply_path.exists():
        needs_refresh = not cache_ply_path.exists() or cache_ply_path.stat().st_mtime < external_ply_path.stat().st_mtime
        if needs_refresh:
            cache_ply_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Copying competition points cache from {external_ply_path} -> {cache_ply_path}")
            shutil.copy2(external_ply_path, cache_ply_path)
        point_count = count_ply_vertices(cache_ply_path)
        return cache_ply_path, point_count, "ply"

    raise FileNotFoundError(
        f"Neither points3D.txt nor points3D.ply was found under {sparse_dir}."
    )
