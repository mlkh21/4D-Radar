import os
import scipy.io as scio 
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import bisect
from abc import ABC, abstractmethod # abstractmethod 用于定义抽象方法（即必须在子类中实现的方法）

class BaseRadarDataset(Dataset, ABC):
    """雷达数据集基类"""
    
    def __init__(self, transform=None):
        self.transform = transform
        self._data = []
    
    def __len__(self):
        return len(self._data)
    
    def apply_transform(self, data):
        """应用变换"""
        if self.transform:
            return self.transform(data)
        return data
    
    @abstractmethod 
    def load_item(self, index):
        """子类实现具体的加载逻辑"""
        raise NotImplementedError
    
    def __getitem__(self, index):
        data = self.load_item(index)
        return self.apply_transform(data)


def load_data_NTU4DRadLM(radarpath, lidarpath, seqname):
    """
    输入:
        radarpath: 雷达数据路径。
        lidarpath: 激光雷达数据路径。
        seqname: 序列名称。
    输出:
        tuple: (雷达图像列表, 激光雷达图像列表, 名称列表)。
    作用: 加载 NTU4DRadLM 数据。
    逻辑:
    1. 读取雷达图像并转换为灰度图。
    2. 读取激光雷达图像并转换为 RGB 图。
    3. 生成名称列表。
    """
    print("seqname", seqname) 
    files = os.listdir(radarpath) 
    files.sort() 
    radar = [] 
    for i in files: 
        path = os.path.join(radarpath, i) 
        radar_img = Image.open(path).convert('L') 
        radar.append(radar_img) 

    files = os.listdir(lidarpath) 
    files.sort() 
    lidar = [] 
    for i in files: 
        path = os.path.join(lidarpath, i) 
        lidar_img = Image.open(path).convert('RGB') 
        lidar.append(lidar_img) 

    name_list = [] 
    for i in files: 
        name_list.append(seqname + "_" + i.split('.')[0]) 
    
    assert(len(radar)==len(lidar)==len(name_list)) 

    return radar, lidar, name_list 

def load_data_benchmark(adcpath, datapath, labelpath, seqname):
    """
    输入:
        adcpath: ADC 数据路径。
        datapath: 数据路径。
        labelpath: 标签路径。
        seqname: 序列名称。
    输出:
        tuple: (ADC 数据列表, 数据列表, 标签列表, 名称列表)。
    作用: 加载基准数据。
    逻辑:
    1. 加载 ADC 数据并重塑形状。
    2. 加载数据并归一化。
    3. 加载标签（如果存在）。
    4. 生成名称列表。
    """
    norm = lambda x: (x - x.min())/(x.max() - x.min()) 

    files = os.listdir(adcpath) 
    files.sort() 
    adc = [] 
    for i in files:
        path = os.path.join(adcpath, i)
        adc_i = np.load(path)['rdm_multi'] 
        adc_i = adc_i.reshape((3, 4, 128, 128)) 
        adc.append(adc_i) 
    
    files = os.listdir(datapath) 
    files.sort()
    data = [norm(scio.loadmat(os.path.join(datapath, i))['rdm'][np.newaxis]) for i in files]

    if not labelpath: 
        return data
    
    files = os.listdir(labelpath) 
    files.sort()
    label = [scio.loadmat(os.path.join(labelpath, i))['label'][np.newaxis] for i in files]
    
    assert(len(data)==len(label)==len(adc)) 

    name_list = []
    for i in files:
        name_list.append(seqname + "_" + i.split('.')[0]) 

    return adc, data, label, name_list 

def load_data(datapath, labelpath = None):
    """
    输入:
        datapath: 数据路径。
        labelpath: 标签路径。
    输出:
        tuple: (数据列表, 标签列表)。
    作用: 加载数据。
    逻辑:
    1. 加载数据并归一化。
    2. 加载标签（如果存在）。
    """
    norm = lambda x: (x - x.min())/(x.max() - x.min()) 
    
    files = os.listdir(datapath)
    files.sort()
    data = [norm(scio.loadmat(os.path.join(datapath, i))['rdm'][np.newaxis]) for i in files]

    if not labelpath:
        return data
    
    files = os.listdir(labelpath)
    files.sort()
    label = [scio.loadmat(os.path.join(labelpath, i))['label'][np.newaxis] for i in files]
    
    assert(len(data)==len(label)) 
                      
    return data, label 

def init_dataset(config, dataset_path, transform, mode):
    """
    输入:
        config: 配置对象。
        dataset_path: 数据集路径。
        transform: 变换函数。
        mode: 模式（train 或 test）。
    输出:
        Dataset: 数据集对象。
    作用: 初始化数据集。
    逻辑:
    1. 根据模式选择训练集或测试集。
    2. 遍历序列，加载数据。
    3. 创建 myDataset_coloradar 对象。
    """
    Radar, Lidar, Name = [], [], [] 
    if mode == "train": 
        for i in config.data.train: 
            radar, lidar, name = load_data_coloradar(dataset_path + "/{}/range_azimuth_heatmap/".format(i), dataset_path + "/{}/lidar_pcl_bev_polar_img/".format(i), i)
            Radar += radar 
            Lidar += lidar
            Name += name
        dataset = myDataset_coloradar(Radar, Lidar, Name, transform) 
        print("Using {} to train".format(config.data.train)) 
        print("Train data - {}".format(dataset.len)) 
    elif mode == "test": 
        for i in config.data.test: 
            radar, lidar, name = load_data_coloradar(dataset_path + "/{}/range_azimuth_heatmap/".format(i), dataset_path + "/{}/lidar_pcl_bev_polar_img/".format(i), i)
            Radar += radar
            Lidar += lidar
            Name += name
        dataset = myDataset_coloradar(Radar, Lidar, Name, transform) 
        print("Using {} to test".format(config.data.test))
        print("Test data - {}".format(dataset.len))

    return dataset 

class myDataset_coloradar(BaseRadarDataset):
    def __init__(self, radar, lidar, name, transform):
        """
        初始化 Coloradar 数据集。
        """
        super().__init__(transform)
        self.radar = radar 
        self.lidar = lidar 
        self.name = name 
        self._data = radar # 用于 __len__

    def load_item(self, index):
        radar = self.radar[index] 
        lidar = self.lidar[index] 
        name = self.name[index] 
        return (radar, lidar, name)

    def apply_transform(self, data):
        radar, lidar, name = data
        if self.transform: 
            radar = self.transform(radar) 
            lidar = self.transform(lidar) 
        return (radar, lidar, name) 

class myDataset_adc(BaseRadarDataset): 
    def __init__(self, adc, data, label, name = None): 
        """
        初始化 ADC 数据集。
        """
        super().__init__(transform=None)
        self.data = data
        self.label = label
        self.adc = adc
        self.name = name
        self._data = data

    def load_item(self, index): 
        return self.adc[index], self.data[index], self.label[index], self.name[index] 

class myDataset(BaseRadarDataset):
    def __init__(self, data, label = None):
        """
        初始化数据集。
        """
        super().__init__(transform=None)
        self.data = data
        self.label = label
        self._data = data

    def load_item(self, index):
        if self.label:
            return self.data[index], self.label[index] 
        return self.data[index] 


# --- New Voxelization Logic ---

def voxelize_pcl(pcl, voxel_size, grid_size):
    """
    输入:
        pcl: 点云数据 (N, 4)。
        voxel_size: 体素大小 [vx, vy, vz]。
        grid_size: 网格大小 [gx, gy, gz]。
    输出:
        numpy.ndarray: 体素化后的网格 (3, H, W)。
    作用: 点云体素化。
    逻辑:
    1. 过滤边界外的点。
    2. 计算网格索引。
    3. 统计每个网格的点数、Z轴总和、特征总和。
    4. 计算占用率、平均高度、平均特征。
    # pcl: (N, 4) [x, y, z, intensity/doppler]
    # voxel_size: [vx, vy, vz]
    # grid_size: [gx, gy, gz]
    """
    min_bound = np.array([-50, -50, -5])
    max_bound = np.array([50, 50, 10])
    
    # Filter points
    mask = np.all((pcl[:, :3] >= min_bound) & (pcl[:, :3] < max_bound), axis=1)
    pcl = pcl[mask]
    
    if len(pcl) == 0:
        return np.zeros((3, grid_size[0], grid_size[1]), dtype=np.float32)

    # Calculate grid indices
    indices = ((pcl[:, :3] - min_bound) / voxel_size).astype(np.int32)
    
    # Grid dimensions: H, W
    H, W = grid_size[0], grid_size[1]
    
    # Initialize grid
    grid = np.zeros((3, H, W), dtype=np.float32)
    
    # Flatten 2D indices
    flat_indices = indices[:, 0] * W + indices[:, 1]
    
    # Occupancy (count)
    count_grid = np.zeros((H * W), dtype=np.float32)
    np.add.at(count_grid, flat_indices, 1)
    
    # Sum Z
    z_sum_grid = np.zeros((H * W), dtype=np.float32)
    np.add.at(z_sum_grid, flat_indices, pcl[:, 2])
    
    # Sum Feature
    feat_sum_grid = np.zeros((H * W), dtype=np.float32)
    np.add.at(feat_sum_grid, flat_indices, pcl[:, 3])
    
    # Avoid division by zero
    mask = count_grid > 0
    
    # Channel 0: Log Occupancy
    grid[0].reshape(-1)[mask] = np.log(count_grid[mask] + 1)
    
    # Channel 1: Average Height
    grid[1].reshape(-1)[mask] = z_sum_grid[mask] / count_grid[mask]
    
    # Channel 2: Average Feature
    grid[2].reshape(-1)[mask] = feat_sum_grid[mask] / count_grid[mask]
    
    return grid

def load_data_voxel_seq(radarpath, lidarpath, seqname):
    """
    输入:
        radarpath: 雷达数据路径。
        lidarpath: 激光雷达数据路径。
        seqname: 序列名称。
    输出:
        tuple: (雷达路径列表, 激光雷达路径列表, 名称列表)。
    作用: 加载体素序列数据。
    逻辑:
    1. 读取索引文件。
    2. 如果索引文件不存在，回退到实时匹配。
    3. 根据索引匹配雷达和激光雷达文件。
    """
    print("seqname", seqname)
    
    # Index files are in the parent directory of the data paths (the scene dir)
    scene_dir = os.path.dirname(radarpath)
    
    radar_idx_file = os.path.join(scene_dir, 'radar_index_sequence.txt')
    lidar_idx_file = os.path.join(scene_dir, 'lidar_index_sequence.txt')
    
    if not os.path.exists(radar_idx_file) or not os.path.exists(lidar_idx_file):
        # Fallback to on-the-fly matching if index files don't exist (or raise error)
        print(f"Warning: Index files not found in {scene_dir}, falling back to on-the-fly matching.")
        return load_data_voxel_seq_onthefly(radarpath, lidarpath, seqname)
        
    with open(radar_idx_file, 'r') as f:
        radar_indices = [int(line.strip()) for line in f]
        
    with open(lidar_idx_file, 'r') as f:
        lidar_indices = [int(line.strip()) for line in f]
        
    assert len(radar_indices) == len(lidar_indices)
    
    # Get all files sorted
    all_radar_files = sorted([f for f in os.listdir(radarpath) if f.endswith('.npy')])
    all_lidar_files = sorted([f for f in os.listdir(lidarpath) if f.endswith('.npy')])
    
    radar_paths = []
    lidar_paths = []
    name_list = []
    
    for r_idx, l_idx in zip(radar_indices, lidar_indices):
        if r_idx >= len(all_radar_files) or l_idx >= len(all_lidar_files):
            continue
            
        r_file = all_radar_files[r_idx]
        l_file = all_lidar_files[l_idx]
        
        radar_paths.append(os.path.join(radarpath, r_file))
        lidar_paths.append(os.path.join(lidarpath, l_file))
        name_list.append(f"{seqname}_{l_file[:-4]}")
            
    return radar_paths, lidar_paths, name_list

def load_data_voxel_seq_onthefly(radarpath, lidarpath, seqname):
    """
    输入:
        radarpath: 雷达数据路径。
        lidarpath: 激光雷达数据路径。
        seqname: 序列名称。
    输出:
        tuple: (雷达路径列表, 激光雷达路径列表, 名称列表)。
    作用: 实时匹配雷达和激光雷达数据。
    逻辑:
    1. 获取所有文件并提取时间戳。
    2. 遍历激光雷达时间戳，寻找最近的雷达时间戳。
    3. 如果时间差小于阈值，则匹配成功。
    """
    # 输入

    radar_files = sorted([f for f in os.listdir(radarpath) if f.endswith('.npy')])
    lidar_files = sorted([f for f in os.listdir(lidarpath) if f.endswith('.npy')])
    
    radar_timestamps = [float(f[:-4]) for f in radar_files]
    lidar_timestamps = [float(f[:-4]) for f in lidar_files]
    
    radar_paths = []
    lidar_paths = []
    name_list = []
    
    for i, l_ts in enumerate(lidar_timestamps):
        idx = bisect.bisect_left(radar_timestamps, l_ts)
        if idx == 0:
            r_idx = 0
        elif idx == len(radar_timestamps):
            r_idx = len(radar_timestamps) - 1
        else:
            before = radar_timestamps[idx - 1]
            after = radar_timestamps[idx]
            if after - l_ts < l_ts - before:
                r_idx = idx
            else:
                r_idx = idx - 1
        
        if abs(radar_timestamps[r_idx] - l_ts) < 0.1:
            l_path = os.path.join(lidarpath, lidar_files[i])
            r_path = os.path.join(radarpath, radar_files[r_idx])
            
            lidar_paths.append(l_path)
            radar_paths.append(r_path)
            name_list.append(f"{seqname}_{l_ts:.6f}")
            
    return radar_paths, lidar_paths, name_list

class myDataset_voxel(BaseRadarDataset):
    DEFAULT_VOXEL_SIZE = np.array([0.2, 0.2, 0.2])
    DEFAULT_GRID_SIZE = [500, 500]

    def __init__(self, radar_paths, lidar_paths, names, transform=None, voxel_size=None, grid_size=None):
        """
        初始化体素数据集。
        """
        super().__init__(transform)
        self.radar_paths = radar_paths
        self.lidar_paths = lidar_paths
        self.names = names
        self._data = radar_paths # For __len__
        
        # Voxelization parameters
        self.voxel_size = voxel_size if voxel_size is not None else self.DEFAULT_VOXEL_SIZE
        self.grid_size = grid_size if grid_size is not None else self.DEFAULT_GRID_SIZE
        
    def load_item(self, index):
        r_path = self.radar_paths[index]
        l_path = self.lidar_paths[index]
        name = self.names[index]
        
        r_pcl = np.load(r_path)
        l_pcl = np.load(l_path)
        
        # Ensure 4 channels (x,y,z,feat)
        if r_pcl.shape[1] < 4:
             r_pcl = np.hstack((r_pcl, np.zeros((r_pcl.shape[0], 4 - r_pcl.shape[1]))))
        if l_pcl.shape[1] < 4:
             l_pcl = np.hstack((l_pcl, np.zeros((l_pcl.shape[0], 4 - l_pcl.shape[1]))))
             
        r_vox = voxelize_pcl(r_pcl, self.voxel_size, self.grid_size)
        l_vox = voxelize_pcl(l_pcl, self.voxel_size, self.grid_size)
            
        return (r_vox, l_vox, name)

    def apply_transform(self, data):
        r_vox, l_vox, name = data
        if self.transform:
            r_vox = self.transform(r_vox)
            l_vox = self.transform(l_vox)
        return (r_vox, l_vox, name)

def init_dataset_voxel(config, dataset_path, transform, mode):
    """
    输入:
        config: 配置对象。
        dataset_path: 数据集路径。
        transform: 变换函数。
        mode: 模式（train 或 test）。
    输出:
        Dataset: 数据集对象。
    作用: 初始化体素数据集。
    逻辑:
    1. 根据模式选择序列。
    2. 遍历序列，加载数据路径。
    3. 创建 myDataset_voxel 对象。
    """
    Radar_paths, Lidar_paths, Names = [], [], []
    
    if mode == "train":
        seqs = config.data.train
    elif mode == "test":
        seqs = config.data.test
    else:
        raise ValueError("Unknown mode: {}".format(mode))
        
    for i in seqs:
        r_path = os.path.join(dataset_path, i, "radar_enhanced_pcl")
        l_path = os.path.join(dataset_path, i, "livox_lidar")
        
        r, l, n = load_data_voxel_seq(r_path, l_path, i)
        Radar_paths += r
        Lidar_paths += l
        Names += n
        
    dataset = myDataset_voxel(Radar_paths, Lidar_paths, Names, transform)
    print(f"Using {seqs} to {mode}")
    print(f"{mode.capitalize()} data - {dataset.len}")
    
    return dataset
