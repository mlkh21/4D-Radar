import os
import numpy as np

RAW_DATA_PATH = "/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/NTU4DRadLM_Raw"
CALIB_PATH = "/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/config/calib_radar_to_livox.txt"

def load_calib(calib_file):
    R = np.eye(3)
    T = np.zeros(3)
    if not os.path.exists(calib_file):
        print("Calib file not found")
        return R, T
    with open(calib_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(':')
            if len(parts) < 2: continue
            key = parts[0].strip()
            raw_parts = parts[1].strip().split()
            vals = []
            for x in raw_parts:
                try:
                    vals.append(float(x))
                except ValueError:
                    continue
            if key == 'R':
                R = np.array(vals).reshape(3, 3)
            elif key == 'T':
                T = np.array(vals)
    return R, T

def transform_pcl(pcl, R, T):
    if pcl.shape[0] == 0: return pcl
    xyz = pcl[:, :3]
    xyz_trans = np.dot(xyz, R.T) + T
    pcl_trans = pcl.copy()
    pcl_trans[:, :3] = xyz_trans
    return pcl_trans

def inspect_scene(scene_name):
    scene_path = os.path.join(RAW_DATA_PATH, scene_name)
    radar_path = os.path.join(scene_path, "radar_pcl")
    
    if not os.path.exists(radar_path):
        radar_path = os.path.join(scene_path, "radar_enhanced_pcl")
        if not os.path.exists(radar_path):
            return

    files = sorted([f for f in os.listdir(radar_path) if f.endswith('.npy')])
    
    R, T = load_calib(CALIB_PATH)
    print(f"R:\n{R}")
    print(f"T: {T}")

    for i in range(min(5, len(files))):
        file_path = os.path.join(radar_path, files[i])
        data = np.load(file_path)
        print(f"File {i}: {files[i]}")
        
        # Transform
        data_trans = transform_pcl(data, R, T)
        
        # Check filtering condition on TRANSFORMED data
        r = np.sqrt(data_trans[:, 0]**2 + data_trans[:, 1]**2)
        azimuth = np.arctan2(data_trans[:, 1], data_trans[:, 0])
        keep = (r < 250.0) & (np.abs(azimuth) < np.pi/2)
        
        print(f"  Original Points: {data.shape[0]}")
        print(f"  Points kept after transform & filter: {np.sum(keep)} / {data.shape[0]}")
        if np.sum(keep) == 0:
            print(f"  Max Azimuth: {np.max(np.abs(azimuth)):.4f} (Limit: {np.pi/2:.4f})")
            print(f"  Min/Max X: {data_trans[:, 0].min():.2f} / {data_trans[:, 0].max():.2f}")

scenes = sorted([d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))])
for scene in scenes:
    print(f"Inspecting scene: {scene}")
    inspect_scene(scene)
