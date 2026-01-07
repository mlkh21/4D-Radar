import rosbag
import os
import argparse
import numpy as np
import pandas as pd
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import glob
import re
import cv2

# 尝试导入 Livox 驱动消息类型
try:
    import sys
    sys.path.append('/home/zxj/catkin_ws/devel/lib/python3/dist-packages')
    from livox_ros_driver.msg import CustomMsg
    LIVOX_AVAILABLE = True
except ImportError:
    LIVOX_AVAILABLE = False
    print("[Warning] 'livox_ros_driver' not found. Will attempt generic parsing for LiDAR.")

# 选择想要导出的主题
ALLOWED_TOPICS = {
    "livox/lidar",
    "radar_pcl",
    "radar_enhanced_pcl",
    "thermal_cam/thermal_image/compressed",
    "ublox/fix",
    "ublox/fix_velocity",
    "vectornav/imu",
}


def _topic_key(topic_name: str) -> str:
    return topic_name.strip('/')

def get_scene_name(bag_path):
    """
    从文件名解析场景名称，用于归并分卷。
    规则：提取第一个下划线前的部分。
    例如: 
    loop1_2022-06-03_0.bag -> loop1
    carpark_2022-06-03.bag -> carpark
    """
    filename = os.path.basename(bag_path)
    # 简单粗暴：取第一个下划线前的名字作为场景名
    # 如果你的命名规则不同，可以在这里修改
    scene_name = filename.split('_')[0]
    return scene_name

def save_pointcloud(msg, save_dir, timestamp):
    """
    保存点云为 .npy 文件，保留所有关键属性。
    Livox: [x, y, z, reflectivity]
    Radar (PointCloud v1): [x, y, z, intensity, velocity]
    """
    try:
        points_list = []
        
        # 1: Livox CustomMsg
        # 结构: x, y, z, reflectivity, tag, line
        if LIVOX_AVAILABLE and 'CustomMsg' in str(type(msg)):
            for p in msg.points:
                # 保存 x, y, z, reflectivity
                points_list.append([p.x, p.y, p.z, float(p.reflectivity)])
        
        # 2: Standard PointCloud2
        elif hasattr(msg, 'width'): # Duck typing for PointCloud2
            # 尝试读取常用字段
            # 注意：不同雷达的字段名可能不同，这里尝试读取 intensity 和 velocity/doppler
            field_names = [f.name for f in msg.fields]
            read_fields = ['x', 'y', 'z']
            
            if 'intensity' in field_names:
                read_fields.append('intensity')
            elif 'reflectivity' in field_names:
                read_fields.append('reflectivity')
            else:
                read_fields.append(None) # 占位

            # 尝试读取速度字段
            if 'velocity' in field_names:
                read_fields.append('velocity')
            elif 'doppler' in field_names:
                read_fields.append('doppler')
            else:
                read_fields.append(None) # 占位
            
            # 过滤掉 None
            actual_fields = [f for f in read_fields if f is not None]
            
            gen = pc2.read_points(msg, field_names=actual_fields, skip_nans=True)
            
            # 统一填充为 5 维: [x, y, z, intensity, velocity]
            # 如果原始数据缺失某维，填 0
            for p in gen:
                point_data = list(p)
                # 补齐维度逻辑... 这里简化处理，直接存原始读取到的数据
                # 后续处理时根据 shape 判断
                points_list.append(point_data)

        # 3: Standard PointCloud (v1) - 常见于 4D Radar
        # 结构: points[], channels[name, values[]]
        elif hasattr(msg, 'points') and msg.points and hasattr(msg.points[0], 'x'):
            # 提取 channels 数据
            channel_map = {}
            for channel in msg.channels:
                channel_map[channel.name.lower()] = channel.values
            
            # 查找强度和速度对应的 channel
            # 常见名: intensity, power, rcs, snr
            # 常见名: velocity, doppler, v_r
            
            intensities = None
            velocities = None
            
            for key in channel_map:
                if any(k in key for k in ['intensity', 'power', 'rcs', 'snr']):
                    intensities = channel_map[key]
                if any(k in key for k in ['velocity', 'doppler', 'v_r']):
                    velocities = channel_map[key]
            
            num_points = len(msg.points)
            for i in range(num_points):
                p = msg.points[i]
                # 默认值
                inten = intensities[i] if intensities else 0.0
                vel = velocities[i] if velocities else 0.0
                
                points_list.append([p.x, p.y, p.z, float(inten), float(vel)])
        
        # 4: Fallback / Unknown
        else:
            print(f"[Warning] Unknown pointcloud type: {type(msg)} at {timestamp}")
            return

        if not points_list:
            return

        pc_np = np.array(points_list, dtype=np.float32)
        
        # 保存为 .npy 
        filename_npy = os.path.join(save_dir, f"{timestamp:.6f}.npy")
        np.save(filename_npy, pc_np)
        
        # 可选：同时保存一份仅含 XYZ 的 PCD 用于可视化预览
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
        # filename_pcd = os.path.join(save_dir, f"{timestamp:.6f}.pcd")
        # o3d.io.write_point_cloud(filename_pcd, pcd)
        
    except Exception as e:
        # 打印错误信息
        print(f"[Error] Failed to save pointcloud at {timestamp}: {e}")
        pass


def save_compressed_image(msg, save_dir, timestamp):
    try:
        data_np = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[Warning] Failed to decode image at {timestamp}")
            return
        filename = os.path.join(save_dir, f"{timestamp:.6f}.png")
        cv2.imwrite(filename, img)
    except Exception as e:
        print(f"[Error] Failed to save image at {timestamp}: {e}")
        pass

def process_ntu_dataset(input_root, output_root):
    # 1. 递归查找所有 .bag 文件
    search_pattern = os.path.join(input_root, "**", "*.bag")
    bag_files = glob.glob(search_pattern, recursive=True)
    bag_files.sort()

    if not bag_files:
        print(f"No .bag files found in {input_root}")
        return

    print(f"Found {len(bag_files)} bag files. Starting processing...")
    print(f"Output Root: {output_root}\n")

    # 2. 遍历处理每个 Bag
    for i, bag_path in enumerate(bag_files):
        # 确定该 bag 属于哪个场景 (loop1, carpark, etc.)
        scene_name = get_scene_name(bag_path)
        
        # 构建最终输出路径: ./output/loop1/
        scene_output_dir = os.path.join(output_root, scene_name)
        
        bag_filename = os.path.basename(bag_path)
        print(f"[{i+1}/{len(bag_files)}] Processing: {bag_filename}")
        print(f"      -> Merging into: {scene_output_dir}")
        
        try:
            bag = rosbag.Bag(bag_path)
        except Exception as e:
            print(f"      [ERROR] Could not open bag: {e}")
            continue

        info = bag.get_type_and_topic_info()
        csv_buffers = {} # 用于缓存非点云数据
        
        for topic, msg, t in bag.read_messages():
            if topic not in info.topics:
                continue

            topic_key = _topic_key(topic)
            if topic_key not in ALLOWED_TOPICS:
                continue
            
            msg_type = info.topics[topic].msg_type
            timestamp = t.to_sec()

            # 准备话题对应的子文件夹名，例如 /livox/lidar -> livox_lidar
            topic_clean = topic.strip('/').replace('/', '_')
            topic_dir = os.path.join(scene_output_dir, topic_clean)

            # A. 处理点云 (LiDAR & Radar)
            if 'PointCloud' in msg_type or 'CustomMsg' in msg_type:
                os.makedirs(topic_dir, exist_ok=True)
                save_pointcloud(msg, topic_dir, timestamp)

            # A2. 处理压缩图像
            elif 'CompressedImage' in msg_type:
                os.makedirs(topic_dir, exist_ok=True)
                save_compressed_image(msg, topic_dir, timestamp)

            # B. 处理其他数据 (IMU, GPS, etc.) -> 存 CSV
            else:
                if topic not in csv_buffers:
                    csv_buffers[topic] = []

                data_row = {'timestamp': timestamp}

                if hasattr(msg, 'header'):
                    data_row['seq'] = msg.header.seq

                # 使用 __slots__ 获取真实数据字段，避免获取到方法(method)
                # 如果没有 __slots__ (极少数情况)，回退到 dir()
                slots = getattr(msg, '__slots__', dir(msg))
                
                for attr in slots:
                    if attr.startswith('_') or attr == 'header':
                        continue
                        
                    try:
                        val = getattr(msg, attr)
                        
                        # 1. 处理嵌套的几何消息 (Vector3, Quaternion) -> 扁平化展开
                        # 检查是否具有 x, y, z 属性
                        if all(hasattr(val, k) for k in ['x', 'y', 'z']):
                            data_row[f"{attr}_x"] = val.x
                            data_row[f"{attr}_y"] = val.y
                            data_row[f"{attr}_z"] = val.z
                            if hasattr(val, 'w'): # Quaternion
                                data_row[f"{attr}_w"] = val.w
                        
                        # 2. 处理列表/数组 (Covariance) -> 转字符串，避免多列混乱
                        elif isinstance(val, (list, tuple, np.ndarray)):
                            data_row[attr] = str(list(val))
                            
                        # 3. 处理基本类型
                        elif isinstance(val, (int, float, str, bool)):
                            data_row[attr] = val
                            
                        # 4. 其他情况 (转字符串)
                        else:
                            # 过滤掉方法(method)
                            if not callable(val):
                                data_row[attr] = str(val).replace('\n', ' ')
                                
                    except Exception as e:
                        # print(f"[Warning] Failed to read attribute {attr} on {topic_clean}: {e}")
                        pass

                csv_buffers[topic].append(data_row)

        bag.close()

        # C. 保存 CSV 数据
        # 为了防止不同分卷覆盖同名 csv，在 csv 文件名中加上分卷标识
        # 例如: vectornav_imu/data_loop1_2022-06-03_0.csv
        bag_base_name = os.path.splitext(bag_filename)[0]
        
        for topic, data_list in csv_buffers.items():
            if not data_list: continue
            
            topic_clean = topic.strip('/').replace('/', '_')
            topic_dir = os.path.join(scene_output_dir, topic_clean)
            os.makedirs(topic_dir, exist_ok=True)
            
            df = pd.DataFrame(data_list)
            # 将时间戳移到第一列
            if 'timestamp' in df.columns:
                cols = ['timestamp'] + [c for c in df.columns if c != 'timestamp']
                df = df[cols]
            
            csv_filename = f"data_{bag_base_name}.csv"
            df.to_csv(os.path.join(topic_dir, csv_filename), index=False)
            
    print("\nAll extraction tasks completed!")

if __name__ == "__main__":
    # 在这里修改默认路径，或者通过命令行参数传入
    default_input = "./NTU4DRadLM_pre_processing/NTU4DRadLM"
    default_output = "./NTU4DRadLM_pre_processing/NTU4DRadLM_Raw"
    
    parser = argparse.ArgumentParser(description="Unpack NTU4DRadLM Bags (No Visual/Thermal)")
    parser.add_argument("--input", default=default_input, help="Input dataset root directory")
    parser.add_argument("--output", default=default_output, help="Output directory")
    
    args = parser.parse_args()
    
    process_ntu_dataset(args.input, args.output)