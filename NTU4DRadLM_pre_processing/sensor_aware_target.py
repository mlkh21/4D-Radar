from dataclasses import dataclass
import argparse
import json
import os
import shutil
from typing import Dict, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np


@dataclass(frozen=True)
class SensorAwareTargetPolicy:
    pc_range: Tuple[float, float, float, float, float, float]
    z_min: Optional[float] = None
    x_max: Optional[float] = None
    require_radar_visibility: bool = False
    radar_visibility_radius: int = 1
    doppler_radius: int = 1


def _voxel_centers(shape: Sequence[int], pc_range: Sequence[float]):
    nx, ny, nz = shape
    x_size = (pc_range[3] - pc_range[0]) / float(nx)
    y_size = (pc_range[4] - pc_range[1]) / float(ny)
    z_size = (pc_range[5] - pc_range[2]) / float(nz)
    x = pc_range[0] + (np.arange(nx, dtype=np.float32) + 0.5) * x_size
    y = pc_range[1] + (np.arange(ny, dtype=np.float32) + 0.5) * y_size
    z = pc_range[2] + (np.arange(nz, dtype=np.float32) + 0.5) * z_size
    return x, y, z


def _dilate_boolean(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()

    result = np.zeros_like(mask, dtype=bool)
    nx, ny, nz = mask.shape
    for x, y, z in np.argwhere(mask):
        x0, x1 = max(0, x - radius), min(nx, x + radius + 1)
        y0, y1 = max(0, y - radius), min(ny, y + radius + 1)
        z0, z1 = max(0, z - radius), min(nz, z + radius + 1)
        result[x0:x1, y0:y1, z0:z1] = True
    return result


def _doppler_from_local_radar(
    lidar_occ: np.ndarray,
    radar_occ: np.ndarray,
    radar_doppler: np.ndarray,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    doppler = np.zeros_like(radar_doppler, dtype=np.float32)
    doppler_mask = np.zeros_like(radar_doppler, dtype=np.float32)
    nx, ny, nz = lidar_occ.shape

    for x, y, z in np.argwhere(lidar_occ):
        x0, x1 = max(0, x - radius), min(nx, x + radius + 1)
        y0, y1 = max(0, y - radius), min(ny, y + radius + 1)
        z0, z1 = max(0, z - radius), min(nz, z + radius + 1)
        local_mask = radar_occ[x0:x1, y0:y1, z0:z1]
        if not np.any(local_mask):
            continue
        local_doppler = radar_doppler[x0:x1, y0:y1, z0:z1][local_mask]
        doppler[x, y, z] = float(np.mean(local_doppler))
        doppler_mask[x, y, z] = 1.0

    return doppler, doppler_mask


def build_sensor_aware_target(
    lidar_voxel: np.ndarray,
    radar_voxel: np.ndarray,
    policy: SensorAwareTargetPolicy,
) -> np.ndarray:
    if lidar_voxel.shape != radar_voxel.shape:
        raise ValueError(f"Shape mismatch: lidar={lidar_voxel.shape}, radar={radar_voxel.shape}")
    if lidar_voxel.ndim != 4 or lidar_voxel.shape[-1] < 4:
        raise ValueError(f"Expected voxel shape (X, Y, Z, C>=4), got {lidar_voxel.shape}")

    target = np.zeros_like(lidar_voxel, dtype=np.float32)
    lidar_occ = lidar_voxel[..., 0] > 0
    keep = lidar_occ.copy()

    x_centers, _, z_centers = _voxel_centers(lidar_voxel.shape[:3], policy.pc_range)
    if policy.z_min is not None:
        keep &= z_centers[None, None, :] >= float(policy.z_min)
    if policy.x_max is not None:
        keep &= x_centers[:, None, None] <= float(policy.x_max)

    radar_occ = radar_voxel[..., 0] > 0
    if policy.require_radar_visibility:
        visible = _dilate_boolean(radar_occ, int(policy.radar_visibility_radius))
        keep &= visible

    target[..., 0] = keep.astype(np.float32)
    target[..., 1] = np.where(keep, lidar_voxel[..., 1], 0.0).astype(np.float32)

    doppler, doppler_mask = _doppler_from_local_radar(
        keep,
        radar_occ,
        radar_voxel[..., 2],
        int(policy.doppler_radius),
    )
    target[..., 2] = doppler
    target[..., 3] = doppler_mask
    return target


def load_voxel(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        voxel = np.zeros(data["shape"], dtype=np.float32)
        coords = data["coords"]
        if coords.shape[0] > 0:
            voxel[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
        return voxel
    return np.load(path).astype(np.float32)


def save_voxel(path: str, voxel: np.ndarray) -> None:
    if path.endswith(".npz"):
        occupied = voxel[..., 0] > 0
        coords = np.column_stack(np.where(occupied))
        features = voxel[occupied]
        np.savez_compressed(path, coords=coords, features=features, shape=voxel.shape)
        return
    np.save(path, voxel.astype(np.float32))


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _matching_file(folder: str, name: str) -> Optional[str]:
    direct = os.path.join(folder, name)
    if os.path.exists(direct):
        return direct
    stem, ext = os.path.splitext(name)
    other_ext = ".npz" if ext == ".npy" else ".npy"
    other = os.path.join(folder, stem + other_ext)
    return other if os.path.exists(other) else None


def build_scene_targets(
    scene_dir: str,
    output_scene_dir: str,
    policy: SensorAwareTargetPolicy,
    max_files: int = 0,
) -> int:
    radar_dir = os.path.join(scene_dir, "radar_voxel")
    lidar_dir = os.path.join(scene_dir, "lidar_voxel")
    out_radar_dir = os.path.join(output_scene_dir, "radar_voxel")
    out_target_dir = os.path.join(output_scene_dir, "target_voxel")
    os.makedirs(out_radar_dir, exist_ok=True)
    os.makedirs(out_target_dir, exist_ok=True)

    radar_files = sorted(
        name for name in os.listdir(radar_dir) if name.endswith((".npy", ".npz"))
    )
    if max_files > 0:
        radar_files = radar_files[: int(max_files)]
    written = 0
    for name in tqdm(radar_files, desc=f"Processing {os.path.basename(scene_dir)}"):
        radar_path = os.path.join(radar_dir, name)
        lidar_path = _matching_file(lidar_dir, name)
        if lidar_path is None:
            continue

        radar = load_voxel(radar_path)
        lidar = load_voxel(lidar_path)
        target = build_sensor_aware_target(lidar, radar, policy)
        out_name = os.path.splitext(name)[0] + os.path.splitext(lidar_path)[1]
        save_voxel(os.path.join(out_target_dir, out_name), target)
        _link_or_copy(radar_path, os.path.join(out_radar_dir, name))
        written += 1

    metadata = {
        "source_scene_dir": os.path.abspath(scene_dir),
        "policy": policy.__dict__,
        "frames_written": written,
    }
    with open(os.path.join(output_scene_dir, "target_policy.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return written


def build_dataset_targets(
    input_root: str,
    output_root: str,
    policy: SensorAwareTargetPolicy,
    scenes: Optional[Sequence[str]] = None,
    max_files: int = 0,
) -> Dict[str, int]:
    if scenes is None:
        scenes = sorted(
            name for name in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, name))
        )

    counts: Dict[str, int] = {}
    for scene in scenes:
        scene_dir = os.path.join(input_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        counts[scene] = build_scene_targets(
            scene_dir,
            os.path.join(output_root, scene),
            policy,
            max_files=max_files,
        )
    return counts


def _parse_scene_list(value: str) -> Optional[Sequence[str]]:
    scenes = [item.strip() for item in value.split(",") if item.strip()]
    return scenes or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sensor-aware target_voxel dataset")
    parser.add_argument("--input_root", default="./Data/NTU4DRadLM_Pre")
    parser.add_argument("--output_root", default="./Data/NTU4DRadLM_Pre_sensor_aware")
    parser.add_argument("--scenes", default="", help="Comma-separated scene names; empty means all scenes")
    parser.add_argument("--pc_range", type=float, nargs=6, default=(0, -20, -6, 120, 20, 10))
    parser.add_argument("--z_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=80.0)
    parser.add_argument("--require_radar_visibility", action="store_true")
    parser.add_argument("--radar_visibility_radius", type=int, default=2)
    parser.add_argument("--doppler_radius", type=int, default=1)
    parser.add_argument("--max_files", type=int, default=0, help="Per-scene file cap; 0 means all")
    args = parser.parse_args()

    policy = SensorAwareTargetPolicy(
        pc_range=tuple(args.pc_range),
        z_min=args.z_min,
        x_max=args.x_max,
        require_radar_visibility=args.require_radar_visibility,
        radar_visibility_radius=args.radar_visibility_radius,
        doppler_radius=args.doppler_radius,
    )
    counts = build_dataset_targets(
        args.input_root,
        args.output_root,
        policy,
        scenes=_parse_scene_list(args.scenes),
        max_files=args.max_files,
    )
    for scene, count in counts.items():
        print(f"{scene}: {count} frames")
    print(f"Saved sensor-aware dataset to: {args.output_root}")


if __name__ == "__main__":
    main()
