#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate an interactive HTML comparison for transformed radar and LiDAR."""

import argparse
import json
import os
from typing import Tuple

import numpy as np


def load_calib(path: str) -> Tuple[np.ndarray, np.ndarray]:
    r_mat = np.eye(3, dtype=np.float32)
    t_vec = np.zeros(3, dtype=np.float32)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) < 2:
                continue
            values = []
            for token in parts[1].split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue
            if parts[0].strip() == "R" and len(values) == 9:
                r_mat = np.asarray(values, dtype=np.float32).reshape(3, 3)
            elif parts[0].strip() == "T" and len(values) >= 3:
                t_vec = np.asarray(values[:3], dtype=np.float32)
    return r_mat, t_vec


def invert_transform(r_mat: np.ndarray, t_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_inv = r_mat.T
    t_inv = -np.dot(r_inv, t_vec)
    return r_inv, t_inv


def rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def transform_points(points: np.ndarray, r_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    return np.dot(xyz, r_mat.T) + t_vec


def read_indices(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def choose_pair(raw_root: str, scene: str, pair_index: int):
    scene_root = os.path.join(raw_root, scene)
    radar_dir = os.path.join(scene_root, "radar_pcl")
    lidar_dir = os.path.join(scene_root, "livox_lidar")
    radar_files = sorted([name for name in os.listdir(radar_dir) if name.endswith(".npy")])
    lidar_files = sorted([name for name in os.listdir(lidar_dir) if name.endswith(".npy")])
    radar_indices = read_indices(os.path.join(scene_root, "radar_index_sequence.txt"))
    lidar_indices = read_indices(os.path.join(scene_root, "lidar_index_sequence.txt"))
    max_pairs = min(len(radar_indices), len(lidar_indices))
    if pair_index < 0 or pair_index >= max_pairs:
        raise IndexError(f"pair_index out of range: {pair_index}, max={max_pairs - 1}")
    radar_idx = radar_indices[pair_index]
    lidar_idx = lidar_indices[pair_index]
    return (
        os.path.join(radar_dir, radar_files[radar_idx]),
        os.path.join(lidar_dir, lidar_files[lidar_idx]),
        radar_files[radar_idx],
        lidar_files[lidar_idx],
    )


def filter_range(points: np.ndarray, pc_range):
    keep = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] <= pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] <= pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] <= pc_range[5])
    )
    return points[keep]


def downsample(points: np.ndarray, max_points: int, seed: int):
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def make_html(radar_xyz, lidar_xyz, meta):
    radar_json = json.dumps(np.round(radar_xyz, 3).tolist(), separators=(",", ":"))
    lidar_json = json.dumps(np.round(lidar_xyz, 3).tolist(), separators=(",", ":"))
    meta_json = json.dumps(meta, ensure_ascii=False)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Radar LiDAR 3D Compare</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #101318; color: #e8edf2; font-family: sans-serif; }}
    #bar {{ position: fixed; top: 0; left: 0; right: 0; padding: 10px 14px; background: rgba(16,19,24,.88); z-index: 2; font-size: 14px; }}
    #hint {{ color: #aab6c2; margin-top: 4px; }}
    canvas {{ width: 100vw; height: 100vh; display: block; }}
    .radar {{ color: #ff5d4d; }}
    .lidar {{ color: #54a8ff; }}
  </style>
</head>
<body>
<div id="bar">
  <div><span class="radar">Radar transformed</span> vs <span class="lidar">LiDAR</span> | <span id="meta"></span></div>
  <div id="hint">Drag to rotate, wheel to zoom, double click to reset. X/Y/Z axes are drawn in white/green/yellow.</div>
</div>
<canvas id="c"></canvas>
<script>
const radar = {radar_json};
const lidar = {lidar_json};
const meta = {meta_json};
document.getElementById('meta').textContent = `${{meta.scene}} pair=${{meta.pair_index}} radar=${{meta.radar_file}} lidar=${{meta.lidar_file}} mode=${{meta.transform_mode}}`;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let yaw = -0.55, pitch = 0.45, zoom = 7.0;
let dragging = false, lastX = 0, lastY = 0;
const all = radar.concat(lidar);
const center = all.reduce((a,p)=>[a[0]+p[0],a[1]+p[1],a[2]+p[2]],[0,0,0]).map(v=>v/all.length);
function resize() {{ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; draw(); }}
function project(p) {{
  let x = p[0]-center[0], y = p[1]-center[1], z = p[2]-center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x1 = cy*x + sy*y, y1 = -sy*x + cy*y, z1 = z;
  let y2 = cp*y1 - sp*z1, z2 = sp*y1 + cp*z1;
  const scale = zoom * canvas.height / 120;
  return [canvas.width/2 + x1*scale, canvas.height/2 - z2*scale, y2];
}}
function drawCloud(points, color, size) {{
  const projected = points.map(p => project(p)).sort((a,b)=>a[2]-b[2]);
  ctx.fillStyle = color;
  for (const p of projected) {{
    ctx.globalAlpha = 0.72;
    ctx.fillRect(p[0], p[1], size, size);
  }}
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
  drawCloud(lidar, '#54a8ff', 1.4 * devicePixelRatio);
  drawCloud(radar, '#ff5d4d', 2.0 * devicePixelRatio);
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
addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interactive raw radar/lidar comparison HTML")
    parser.add_argument("--raw_root", default="Data/NTU4DRadLM_Raw")
    parser.add_argument("--scene", default="loop3")
    parser.add_argument("--pair_index", type=int, default=0)
    parser.add_argument("--calib", default="Data/config/calib_radar_to_livox.txt")
    parser.add_argument("--transform_mode", choices=["forward", "inverse", "none"], default="forward")
    parser.add_argument("--extra_roll_deg", type=float, default=0.0, help="Extra rotation around x axis after calib")
    parser.add_argument("--pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10])
    parser.add_argument("--max_radar_points", type=int, default=4000)
    parser.add_argument("--max_lidar_points", type=int, default=12000)
    parser.add_argument("--output", default="Result/alignment_check/loop3/raw_compare_interactive.html")
    return parser.parse_args()


def main():
    args = parse_args()
    radar_path, lidar_path, radar_file, lidar_file = choose_pair(args.raw_root, args.scene, args.pair_index)
    radar = np.load(radar_path).astype(np.float32)
    lidar = np.load(lidar_path).astype(np.float32)
    r_mat, t_vec = load_calib(args.calib)
    if args.transform_mode == "inverse":
        r_mat, t_vec = invert_transform(r_mat, t_vec)
    if abs(args.extra_roll_deg) > 1e-8:
        r_extra = rotation_x(args.extra_roll_deg)
        r_mat = np.dot(r_extra, r_mat)
        t_vec = np.dot(r_extra, t_vec)
    if args.transform_mode == "none":
        radar_xyz = radar[:, :3]
    else:
        radar_xyz = transform_points(radar, r_mat, t_vec)
    lidar_xyz = lidar[:, :3]
    radar_xyz = filter_range(radar_xyz, args.pc_range)
    lidar_xyz = filter_range(lidar_xyz, args.pc_range)
    radar_xyz = downsample(radar_xyz, args.max_radar_points, 1)
    lidar_xyz = downsample(lidar_xyz, args.max_lidar_points, 2)
    meta = {
        "scene": args.scene,
        "pair_index": args.pair_index,
        "radar_file": radar_file,
        "lidar_file": lidar_file,
        "transform_mode": args.transform_mode,
        "extra_roll_deg": args.extra_roll_deg,
        "radar_points": int(radar_xyz.shape[0]),
        "lidar_points": int(lidar_xyz.shape[0]),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(make_html(radar_xyz, lidar_xyz, meta))
    print(f"Saved interactive comparison to: {args.output}")
    print(f"Radar points: {radar_xyz.shape[0]}, LiDAR points: {lidar_xyz.shape[0]}")


if __name__ == "__main__":
    main()
