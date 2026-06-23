#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate interactive 3D HTML comparisons for mini inference outputs."""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)


def sparse_npz_to_points(path: str, pc_range: Sequence[float] = DEFAULT_PC_RANGE) -> np.ndarray:
    data = np.load(path)
    coords = data["coords"].astype(np.float32)
    features = data["features"].astype(np.float32)
    shape = tuple(int(v) for v in data["shape"])
    if len(shape) != 4:
        raise ValueError(f"Expected sparse shape [X,Y,Z,C], got {shape} from {path}")
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    nx, ny, nz = shape[:3]
    xyz = np.empty((coords.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = x_min + (coords[:, 0] + 0.5) * ((x_max - x_min) / max(nx, 1))
    xyz[:, 1] = y_min + (coords[:, 1] + 0.5) * ((y_max - y_min) / max(ny, 1))
    xyz[:, 2] = z_min + (coords[:, 2] + 0.5) * ((z_max - z_min) / max(nz, 1))
    if features.shape[1] > 0:
        return np.concatenate([xyz, features[:, :1]], axis=1)
    return xyz


def load_pred_points(path: str) -> np.ndarray:
    pts = np.load(path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected point array [N,>=3], got {pts.shape} from {path}")
    return pts[:, :4] if pts.shape[1] >= 4 else pts[:, :3]


def resolve_raw_lidar_path(frame_id: str, raw_lidar_dir: str, lidar_index_file: str) -> str:
    """Resolve a processed frame to its timestamped raw LiDAR file."""
    frame_index = int(frame_id)
    with open(lidar_index_file, "r", encoding="utf-8") as handle:
        sequence = [int(line.strip()) for line in handle if line.strip()]
    if frame_index >= len(sequence):
        raise IndexError(f"Frame {frame_id} is outside LiDAR index sequence ({len(sequence)} entries)")

    lidar_files = sorted(
        name for name in os.listdir(raw_lidar_dir) if name.lower().endswith(".npy")
    )
    raw_index = sequence[frame_index]
    if raw_index < 0 or raw_index >= len(lidar_files):
        raise IndexError(f"LiDAR index {raw_index} is outside raw directory ({len(lidar_files)} files)")
    return os.path.join(raw_lidar_dir, lidar_files[raw_index])


def load_raw_lidar_points(path: str) -> np.ndarray:
    points = np.load(path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected raw LiDAR array [N,>=3], got {points.shape} from {path}")
    keep = np.isfinite(points[:, :3]).all(axis=1) & (np.linalg.norm(points[:, :3], axis=1) > 1e-6)
    points = points[keep]
    return points[:, :4] if points.shape[1] >= 4 else points[:, :3]


def filter_points(points: np.ndarray, pc_range: Sequence[float], z_min: float = None, x_max: float = None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts.reshape(0, pts.shape[-1] if pts.ndim == 2 else 3)
    keep = (
        (pts[:, 0] >= float(pc_range[0]))
        & (pts[:, 0] <= float(pc_range[3] if x_max is None else x_max))
        & (pts[:, 1] >= float(pc_range[1]))
        & (pts[:, 1] <= float(pc_range[4]))
        & (pts[:, 2] >= float(pc_range[2] if z_min is None else z_min))
        & (pts[:, 2] <= float(pc_range[5]))
    )
    return pts[keep]


def downsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def cloud_stats(points: np.ndarray) -> Dict[str, object]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return {"count": 0, "centroid": None, "min": None, "max": None}
    xyz = pts[:, :3]
    return {
        "count": int(pts.shape[0]),
        "centroid": np.round(xyz.mean(axis=0), 3).tolist(),
        "min": np.round(xyz.min(axis=0), 3).tolist(),
        "max": np.round(xyz.max(axis=0), 3).tolist(),
    }


def point_json(points: np.ndarray) -> str:
    return json.dumps(np.round(points[:, :3], 3).tolist(), separators=(",", ":"))


def make_html(clouds: List[Dict[str, object]], meta: Dict[str, object]) -> str:
    clouds_js = []
    for cloud in clouds:
        clouds_js.append(
            {
                "name": cloud["name"],
                "color": cloud["color"],
                "size": cloud["size"],
                "points": json.loads(point_json(cloud["points"])),
                "stats": cloud["stats"],
            }
        )
    clouds_json = json.dumps(clouds_js, ensure_ascii=False, separators=(",", ":"))
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Raw LiDAR Inference 3D Compare</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #101318; color: #eef3f8; font-family: Arial, sans-serif; }}
    #bar {{ position: fixed; top: 0; left: 0; right: 0; padding: 10px 14px; background: rgba(16,19,24,.92); z-index: 2; font-size: 14px; }}
    #legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px; }}
    #panel {{ position: fixed; right: 12px; top: 96px; width: 360px; max-height: calc(100vh - 120px); overflow: auto; background: rgba(16,19,24,.82); border: 1px solid #334; padding: 10px; font: 12px/1.4 monospace; white-space: pre-wrap; }}
    canvas {{ width: 100vw; height: 100vh; display: block; }}
    label {{ user-select: none; }}
  </style>
</head>
<body>
<div id="bar">
  <div><b>Radar / Raw LiDAR / LDM / CD interactive comparison</b> | frame <span id="frame"></span></div>
  <div id="legend"></div>
  <div>Drag rotate, wheel zoom, double click reset. Axes: X white, Y green, Z yellow.</div>
</div>
<pre id="panel"></pre>
<canvas id="c"></canvas>
<script>
const clouds = {clouds_json};
const meta = {meta_json};
document.getElementById('frame').textContent = meta.frame_id;
const legend = document.getElementById('legend');
for (const c of clouds) {{
  const item = document.createElement('label');
  item.innerHTML = `<input type="checkbox" checked data-name="${{c.name}}"> <span style="color:${{c.color}}">${{c.name}}</span> (${{c.stats.count}})`;
  legend.appendChild(item);
}}
document.getElementById('panel').textContent = JSON.stringify(meta, null, 2) + "\\n\\n" + JSON.stringify(clouds.map(c => ({{name:c.name, stats:c.stats}})), null, 2);
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let yaw = -0.55, pitch = 0.45, zoom = 7.0;
let dragging = false, lastX = 0, lastY = 0;
const all = clouds.flatMap(c => c.points);
const center = all.length ? all.reduce((a,p)=>[a[0]+p[0],a[1]+p[1],a[2]+p[2]],[0,0,0]).map(v=>v/all.length) : [40,0,0];
function visible(name) {{
  const el = document.querySelector(`input[data-name="${{name}}"]`);
  return el && el.checked;
}}
function resize() {{ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; draw(); }}
function project(p) {{
  let x = p[0]-center[0], y = p[1]-center[1], z = p[2]-center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x1 = cy*x + sy*y, y1 = -sy*x + cy*y, z1 = z;
  let y2 = cp*y1 - sp*z1, z2 = sp*y1 + cp*z1;
  const scale = zoom * canvas.height / 120;
  return [canvas.width/2 + x1*scale, canvas.height/2 - z2*scale, y2];
}}
function drawCloud(cloud) {{
  if (!visible(cloud.name)) return;
  const projected = cloud.points.map(p => project(p)).sort((a,b)=>a[2]-b[2]);
  ctx.fillStyle = cloud.color;
  ctx.globalAlpha = cloud.name === 'raw_lidar' ? 0.55 : 0.76;
  const size = cloud.size * devicePixelRatio;
  for (const p of projected) ctx.fillRect(p[0], p[1], size, size);
  ctx.globalAlpha = 1;
}}
function drawAxis() {{
  const axes = [[[0,0,0],[10,0,0],'#ffffff','X'], [[0,0,0],[0,10,0],'#51d88a','Y'], [[0,0,0],[0,0,5],'#ffe066','Z']];
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.font = `${{13 * devicePixelRatio}}px sans-serif`;
  for (const [a,b,color,label] of axes) {{
    const pa = project([center[0]+a[0], center[1]+a[1], center[2]+a[2]]);
    const pb = project([center[0]+b[0], center[1]+b[1], center[2]+b[2]]);
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    ctx.fillText(label, pb[0]+5, pb[1]-5);
  }}
}}
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#101318'; ctx.fillRect(0,0,canvas.width,canvas.height);
  drawAxis();
  for (const cloud of clouds) drawCloud(cloud);
}}
canvas.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
addEventListener('mouseup', () => dragging = false);
addEventListener('mousemove', e => {{
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.006;
  pitch += (e.clientY - lastY) * 0.006;
  pitch = Math.max(-1.5, Math.min(1.5, pitch));
  lastX = e.clientX; lastY = e.clientY; draw();
}});
canvas.addEventListener('wheel', e => {{ e.preventDefault(); zoom *= Math.exp(-e.deltaY * 0.001); zoom = Math.max(1, Math.min(40, zoom)); draw(); }}, {{passive:false}});
canvas.addEventListener('dblclick', () => {{ yaw = -0.55; pitch = 0.45; zoom = 7.0; draw(); }});
legend.addEventListener('change', draw);
addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def build_frame(args, frame_id: str, meta: Dict[str, object]) -> str:
    radar_path = os.path.join(args.pre_dir, "radar_voxel", f"{frame_id}.npz")
    raw_lidar_path = resolve_raw_lidar_path(frame_id, args.raw_lidar_dir, args.lidar_index_file)
    ldm_path = os.path.join(args.ldm_dir, f"{frame_id}_pcl.npy")
    cd_path = os.path.join(args.cd_dir, f"{frame_id}_pcl.npy")

    radar = sparse_npz_to_points(radar_path, args.pc_range)
    raw_lidar = load_raw_lidar_points(raw_lidar_path)
    ldm = load_pred_points(ldm_path)
    cd = load_pred_points(cd_path)

    if args.z_min is not None or args.x_max is not None:
        radar = filter_points(radar, args.pc_range, args.z_min, args.x_max)
        raw_lidar = filter_points(raw_lidar, args.pc_range, args.z_min, args.x_max)
        ldm = filter_points(ldm, args.pc_range, args.z_min, args.x_max)
        cd = filter_points(cd, args.pc_range, args.z_min, args.x_max)

    clouds = [
        {"name": "raw_lidar", "points": downsample(raw_lidar, args.max_lidar_points, 2), "color": "#54a8ff", "size": 1.3, "stats": cloud_stats(raw_lidar)},
        {"name": "radar", "points": downsample(radar, args.max_radar_points, 1), "color": "#ff5d4d", "size": 2.2, "stats": cloud_stats(radar)},
        {"name": "ldm_pred", "points": downsample(ldm, args.max_pred_points, 3), "color": "#ffcf33", "size": 1.7, "stats": cloud_stats(ldm)},
        {"name": "cd_pred", "points": downsample(cd, args.max_pred_points, 4), "color": "#a678ff", "size": 1.7, "stats": cloud_stats(cd)},
    ]
    frame_meta = dict(meta)
    frame_meta.update(
        {
            "frame_id": frame_id,
            "raw_lidar_file": os.path.basename(raw_lidar_path),
            "z_min_filter": args.z_min,
            "x_max_filter": args.x_max,
        }
    )
    output = os.path.join(args.output_dir, f"raw_lidar_compare_{frame_id}.html")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(make_html(clouds, frame_meta))
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interactive inference comparison HTML")
    parser.add_argument("--pre_dir", default="Data/NTU4DRadLM_Pre/loop3")
    parser.add_argument("--raw_lidar_dir", default="Data/NTU4DRadLM_Raw/loop3/livox_lidar")
    parser.add_argument("--lidar_index_file", default="Data/NTU4DRadLM_Raw/loop3/lidar_index_sequence.txt")
    parser.add_argument("--ldm_dir", default="test/mini-test/inference_results_mini/loop3_ldm_eval")
    parser.add_argument("--cd_dir", default="test/mini-test/inference_results_mini/loop3_cd_eval")
    parser.add_argument("--output_dir", default="Result/visualization/mini_inference_compare")
    parser.add_argument("--frames", default="000068,000150,000253,000386,000478,000488")
    parser.add_argument("--pc_range", type=float, nargs=6, default=list(DEFAULT_PC_RANGE))
    parser.add_argument("--z_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=80.0)
    parser.add_argument("--max_radar_points", type=int, default=4000)
    parser.add_argument("--max_lidar_points", type=int, default=12000)
    parser.add_argument("--max_pred_points", type=int, default=7000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        "pre_dir": args.pre_dir,
        "raw_lidar_dir": args.raw_lidar_dir,
        "ldm_dir": args.ldm_dir,
        "cd_dir": args.cd_dir,
        "pc_range": [float(v) for v in args.pc_range],
    }
    outputs = []
    for frame in [chunk.strip() for chunk in args.frames.split(",") if chunk.strip()]:
        outputs.append(build_frame(args, frame, meta))
    print("Saved interactive comparison HTML files:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
