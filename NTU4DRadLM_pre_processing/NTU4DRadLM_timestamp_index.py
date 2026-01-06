# -*- coding: utf-8 -*-
# 用于生成雷达和激光雷达时间戳的索引文件，以便对齐两种传感器的数据。

import os
import bisect

"""时间戳索引辅助脚本"""

def find_nearest_index(timestamps, target):
    """在有序时间戳中查找与目标最接近的索引。"""
    # timestamps 必须是有序的
    idx = bisect.bisect_left(timestamps, target)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    if after - target < target - before:
        return idx
    else:
        return idx - 1

# 生成新的索引文件
def generate_new_files(directory):
    print("start!")
    # 遍历目录下的每个场景文件夹
    for scene_dir in os.listdir(directory):
        scene_path = os.path.join(directory, scene_dir)
        if not os.path.isdir(scene_path):
            continue
            
        print("Processing scene:", scene_dir)
        
        radar_path = os.path.join(scene_path, 'radar_pcl')
        lidar_path = os.path.join(scene_path, 'livox_lidar')

        if not os.path.exists(radar_path) or not os.path.exists(lidar_path):
            print(f"Skipping {scene_dir}: data directories not found")
            continue

        # 根据文件名提取时间戳
        # 文件名格式示例: 1645868413.022228.npy
        radar_files = sorted([f for f in os.listdir(radar_path) if f.endswith('.npy')])
        lidar_files = sorted([f for f in os.listdir(lidar_path) if f.endswith('.npy')])
        
        timestamps_radar = [float(f.replace('.npy', '')) for f in radar_files]
        timestamps_lidar = [float(f.replace('.npy', '')) for f in lidar_files]

        a = len(timestamps_radar)
        b = len(timestamps_lidar)
        
        print(f"Radar frames: {a}, Lidar frames: {b}")
    
        # 输出文件路径
        new_file_radar = os.path.join(scene_path, 'radar_index_sequence.txt')
        new_file_lidar = os.path.join(scene_path, 'lidar_index_sequence.txt')

        if a <= b:
            print("radar less than or equal to lidar")
            smaller_timestamps = timestamps_radar
            larger_timestamps = timestamps_lidar
            smaller_len = a
            
            # 以Radar为参考（按顺序）
            with open(new_file_radar, 'w') as f_rad, open(new_file_lidar, 'w') as f_lid:
                for i in range(smaller_len):
                    f_rad.write(str(i) + '\n')
                    nearest_idx = find_nearest_index(larger_timestamps, smaller_timestamps[i])
                    f_lid.write(str(nearest_idx) + '\n')
        else:
            print("radar more than lidar")
            smaller_timestamps = timestamps_lidar
            larger_timestamps = timestamps_radar
            smaller_len = b
            # 以Lidar为参考（按顺序）
            with open(new_file_radar, 'w') as f_rad, open(new_file_lidar, 'w') as f_lid:
                for i in range(smaller_len):
                    f_lid.write(str(i) + '\n')
                    nearest_idx = find_nearest_index(larger_timestamps, smaller_timestamps[i])
                    f_rad.write(str(nearest_idx) + '\n')

        print(f"Generated index files for {scene_dir}")

if __name__ == "__main__":
    generate_new_files('/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/NTU4DRadLM_Raw')
