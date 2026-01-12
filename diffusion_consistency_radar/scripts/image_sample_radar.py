# -*- coding: utf-8 -*-
"""
从模型生成大量图像样本，并将它们保存为一个大型 numpy 数组。
这可用于生成用于 FID 评估的样本。
"""

import argparse
# -*- coding: utf-8 -*-
import os
import sys

# 强制将当前项目的路径添加到 sys.path 的最前面，以优先加载当前项目的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util_cond import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from PIL import Image
import torchvision.transforms as transforms
from cm.dataset_loader import NTU4DRadLM_VoxelDataset
import random

seed = 42
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
import yaml
from easydict import EasyDict as edict
import cv2
import open3d as o3d
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Moved to main with args


from pathlib import Path

def get_base_store_path():
    """从环境变量或配置文件获取存储路径"""
    return os.environ.get('RADAR_RESULT_PATH', './diffusion_consistency_radar/results/')

# 全局变量定义
BASE_STORE_PATH = get_base_store_path() # 保存路径
MAX_RANGE = 16.0 # 最大范围(米)
IMAGE_SHAPE = 160 # 图像尺寸(像素)
RANGE_RESOLUTION = 0.125 # 距离分辨率(米/像素)

# 将激光雷达点云转换为鸟瞰图
def pcl_to_cartesian_image (lidar_pcl, save_path):

    # 输入： lidar_pcl: 激光雷达点云数据
    #       save_path: 保存路径
    # 输出： 无，直接保存图像文件

    X_pixel = int((MAX_RANGE - 0)/ RANGE_RESOLUTION) # x方向像素数
    Y_pixel = int((MAX_RANGE - (-MAX_RANGE)) / (RANGE_RESOLUTION * 2)) # y方向像素数
    lidar_bev_image = np.zeros((X_pixel, Y_pixel)) # 初始化BEV图像

    x_grid = np.linspace(0, MAX_RANGE, X_pixel + 1) # x方向网格坐标,linspace(start, stop, num)生成在[start, stop]范围内的num个均匀分布的样本
    # print("x_grid", x_grid)
    y_grid = np.linspace(-MAX_RANGE, MAX_RANGE, Y_pixel + 1) # y方向网格坐标

    if lidar_pcl.shape[0] == 0: # 如果点云为空，直接保存空图像，shape是点云的维度信息
        im = Image.fromarray(np.fliplr((np.flipud(lidar_bev_image)*255)).astype(np.uint8)) # Image.fromarray()将数组转换为图像对象,np.flipud()上下翻转图像，np.fliplr()左右翻转图像
        im = im.resize((IMAGE_SHAPE * 2, IMAGE_SHAPE)) # 调整图像尺寸
        im.save(save_path) # 保存图像文件
        return 
    
    x = lidar_pcl[:, 0] # 提取点云的x坐标
    y = lidar_pcl[:, 1] # 提取点云的y坐标


    # 直接通过公式计算索引，避免循环
    x_indices = ((x - 0) / RANGE_RESOLUTION).astype(int)
    # 注意 y 的范围是从 -MAX_RANGE 开始，且分辨率是 RANGE_RESOLUTION * 2
    y_indices = ((y - (-MAX_RANGE)) / (RANGE_RESOLUTION * 2)).astype(int)
    # 过滤掉超出图像范围的点
    valid_mask = (x_indices >= 0) & (x_indices < X_pixel) & \
                 (y_indices >= 0) & (y_indices < Y_pixel)
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    # 直接利用 numpy 的索引赋值
    lidar_bev_image[x_indices, y_indices] = 1
 
    # for i in range(lidar_pcl.shape[0]): # lidar_pcl.shape[0]是指lidar_pcl的行数
    #     x_diff = np.abs(x_grid - x[i]) # 计算x坐标与网格坐标的差值
    #     y_diff = np.abs(y_grid - y[i]) # 计算y坐标与网格坐标的差值
    #     # print("y_diff", y_grid - y[i])
    #     x_i = np.argmin(x_diff) # 找到最小差值对应的索引
    #     y_i = np.argmin(y_diff) 

    #     if(x_i >= lidar_bev_image.shape[0]) or (y_i >= lidar_bev_image.shape[1]): # 越界检查，如果索引超出图像范围则跳过该点
    #         continue 
    #     lidar_bev_image[x_i,y_i] = 1 # 在BEV图像中标记该点


    # print("lidar_bev_image", lidar_bev_image.shape)
    im = Image.fromarray(np.fliplr((np.flipud(lidar_bev_image)*255)).astype(np.uint8)) # Image.fromarray()将数组转换为图像对象,np.flipud()上下翻转图像，np.fliplr()左右翻转图像
    im = im.resize((IMAGE_SHAPE * 2, IMAGE_SHAPE)) # 调整图像尺寸
    im.save(save_path) # 保存图像文件
    # return im

# 将极坐标图像转换为点云
def polar_image_to_pcl (polar_image):

    # 输入： polar_image: 极坐标图像 (H, W, 3) BGR format if read by cv2
    # 输出： pcl: 点云数据（numpy数组格式）
    #       pcl_o3d: 点云数据（Open3D格式）

    # Handle 3-channel input (Representation Expansion)
    if len(polar_image.shape) == 3:
        # Assuming BGR format from cv2.imread
        # Red (Index 2) = Occupancy
        # Green (Index 1) = Height
        # Blue (Index 0) = Reserved (0)
        occupancy_map = polar_image[:, :, 2]
        height_map = polar_image[:, :, 1]
    else:
        # Fallback for legacy 1-channel
        occupancy_map = polar_image
        height_map = np.zeros_like(polar_image)

    width, height = occupancy_map.shape # Note: width=Rows, height=Cols based on original logic

    point_cloud = []

    # 设定与生成时相同的参数
    Z_MIN = -3.0
    Z_MAX = 7.0
    
    for row in range(height): 
        for column in range(width): 
            
            pixel_val = occupancy_map[column, row]
            
            # 阈值过滤，例如大于 0 或某个噪声阈值
            if pixel_val > 5: 

                column_true = width - column 
                distance = column_true / width * MAX_RANGE 
                
                row_true = height - row 
                angle = (row_true - height / 2) / (height / 2) * (math.pi / 2) 
                
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)

                # 从 Green 通道解码高度
                h_val = height_map[column, row]
                z = (h_val / 255.0) * (Z_MAX - Z_MIN) + Z_MIN

                point_cloud.append([x, y, z])

    pcl = np.array(point_cloud)

    pcl_o3d = o3d.geometry.PointCloud() # 创建Open3D点云对象
    if len(point_cloud) > 0:
        pcl_o3d.points = o3d.utility.Vector3dVector(point_cloud) # 将点云数据转换为Open3D格式
    else:
        pcl_o3d.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    
    return pcl, pcl_o3d

def main():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        attention_resolutions="32,16,8",
        in_ch=8,
        out_ch=3,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        weight_schedule="karras",
        batch_size=4,
        num_workers=4,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--training_mode", type=str, default="edm", choices=["edm", "consistency_distillation"], help="Training mode")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # 设置随机种子
    seed = args.seed
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dist_util.setup_dist() # 设置分布式训练环境
    logger.configure() # 配置日志记录器

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating data loader...")
    # 使用配置中的数据参数

    # 临时从环境变量或默认路径获取，因为 DataConfig 定义里没写 dataset_dir
    dataset_dir = os.environ.get("DATASET_DIR", './NTU4DRadLM_pre_processing/NTU4DRadLM_Pre') 
    
    test_data = NTU4DRadLM_VoxelDataset(
        root_dir=dataset_dir,
        split='test',
        return_path=True,
        alignment_size=32 # 也可以放到 cfg.data.alignment_size
    )

    datal= th.utils.data.DataLoader(
        test_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False)

    data = iter(datal)

    logger.log("creating model and diffusion...")
    
    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_kwargs['distillation'] = distillation
    
    model, diffusion = create_model_and_diffusion(
        **model_kwargs
    )
    
    # ... 后续逻辑保持不变，但要注意 args.sampler 等参数需要从 cfg 或 argparse 获取
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # ...

    # model.to("cpu")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    if not os.path.exists(BASE_STORE_PATH):
        os.mkdir(BASE_STORE_PATH)

    if not os.path.exists(BASE_STORE_PATH + args.output_dir):
        os.mkdir(BASE_STORE_PATH + args.output_dir)
        

    # i = 0

    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)
    cnt = 0
    
    
    for test_i, (b, m, path) in enumerate(datal):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        # b, m, path = next(data)  # 应该从数据加载器 "data" 返回一个图像
        # 在 NTU4DRadLM_VoxelDataset 中: b=target, m=radar, path=path
        
        # c = th.randn_like(b)
        # img = th.cat((b, c), dim=1)     # 添加一个噪声通道

        b = b.to(dist_util.dev())
        m = m.to(dist_util.dev()) # 雷达条件

        radar_condition_dict = {"y": m} # 使用雷达作为条件
        
        # 从批次确定形状
        # b shape: (B, C, Z, H, W)
        sample_shape = (b.shape[0], b.shape[1], b.shape[2], b.shape[3], b.shape[4])

        # start.record()：用于记录时间的开始点
        # sample = karras_sample(...)：调用karras_sample函数生成样本,传入扩散模型、形状、步骤数等参数
        # end.record()：用于记录时间的结束点
        # th.cuda.synchronize()：确保所有CUDA操作完成，防止时间测量不准确
        # print('time for 1 sample', start.elapsed_time(end))：计算并打印生成样本所用的时间
        start.record()
        sample = karras_sample(
            diffusion = diffusion,
            model = model,
            shape = sample_shape,
            steps=args.steps,
            model_kwargs=radar_condition_dict,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
        )

        end.record()
        th.cuda.synchronize()
        print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

        # cnt： 计数器，用于跟踪已处理的样本数量
        cnt = cnt + args.batch_size
        print("cnt", cnt)
        
        # 保存样本
        sample = sample.cpu().numpy()
        for i in range(args.batch_size):
            # path[i] 是目标体素文件的绝对路径
            # 提取场景名称和文件名
            # 例如 .../scene1/target_voxel/0001.npy
            
            file_name = os.path.basename(path[i])
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(path[i])))
            
            save_dir = os.path.join(BASE_STORE_PATH, args.output_dir, scene_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            save_path = os.path.join(save_dir, file_name)
            
            # 保存为 .npy
            np.save(save_path, sample[i])
            print(f"Saved sample to {save_path}")
            polar_image_path = save_path + "/pre_polar_image/"
            cartesian_image_path = save_path + "/pre_cartesian_image/"
            pcl_np_path = save_path + "/pre_pcl_np/"
            pcl_mesh_path = save_path + "/pre_pcl_mesh/"
            gt_polar_path = save_path + "/gt_polar_image/"
            gt_bev_path = save_path + "/gt_bev_image/"
            gt_bev_pcl_path = save_path + "/gt_bev_pcl/"



            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(cartesian_image_path):
                os.mkdir(cartesian_image_path)
            if not os.path.exists(polar_image_path):
                os.mkdir(polar_image_path)
            if not os.path.exists(pcl_np_path):
                os.mkdir(pcl_np_path)    
            if not os.path.exists(pcl_mesh_path):
                os.mkdir(pcl_mesh_path)   
            if not os.path.exists(gt_polar_path):
                os.mkdir(gt_polar_path)   
            if not os.path.exists(gt_bev_path):
                os.mkdir(gt_bev_path)   
            if not os.path.exists(gt_bev_pcl_path):
                os.mkdir(gt_bev_pcl_path)                   

            #save gt
            gt = (m[i] * 255).clamp(0, 255).to(th.uint8)

            gt_numpy = gt.cpu().numpy()
            gt_numpy = np.transpose(gt_numpy, (1, 2, 0))
            gt_numpy_img = Image.fromarray((gt_numpy).astype(np.uint8), mode='RGB')
        

            gt_file_name = gt_polar_path + image_id +'.png'
            gt_numpy_img.save(gt_file_name)


            image_gt = cv2.imread(gt_file_name, cv2.IMREAD_COLOR)

            pcl_gt, pcl_o3d_gt = polar_image_to_pcl(image_gt)
            
            gt_cartesian_save_path = gt_bev_path + image_id +'.png'
            pcl_to_cartesian_image(pcl_gt, gt_cartesian_save_path)
            np.save(gt_bev_pcl_path + image_id +'.npy', pcl_gt)

            #radar prediction

            sample_i = (sample[i] * 255).clamp(0, 255).to(th.uint8)

            sample_i = sample_i.permute(1, 2, 0)
            sample_i = sample_i.contiguous()

            sample_numpy = sample_i.cpu().numpy()
            img_numpy_img = Image.fromarray((sample_numpy).astype(np.uint8), mode='RGB')

            im1_file_name = polar_image_path + image_id +'.png'
            img_numpy_img.save(im1_file_name)

            #save as pcl and mesh
            image_i = cv2.imread(im1_file_name, cv2.IMREAD_COLOR)

            pcl, pcl_o3d = polar_image_to_pcl(image_i)
            np.save(pcl_np_path + image_id +'.npy', pcl)


            o3d.io.write_point_cloud(pcl_mesh_path + image_id +'.ply', pcl_o3d)
            
            cartesian_save_path = cartesian_image_path + image_id +'.png'
            pcl_to_cartesian_image(pcl, cartesian_save_path)


    dist.barrier()
    logger.log("sampling complete")



def create_argparser(): # 定义命令行参数解析器，判断据不同的参数设置执行不同的操作
    defaults = dict(
        # data_dir="/home/ubuntu/coloradar_ws/consistency_model_coloradar/cmu_dataset/",
        training_mode="edm", # 训练模式，支持"edm"和"consistency_distillation"
        image_size=160,  # 图像尺寸
        use_fp16=False,  # 是否使用半精度浮点数
        sigma_min=0.002,  # 最小噪声标准差
        sigma_max=80,   # 最大噪声标准差
        generator="determ", # 随机数生成器类型
        clip_denoised=True, # 是否裁剪去噪后的图像
        num_samples=100,  # 采样数量
        batch_size=1,  # 批处理大小
        sampler="heun",  # 采样器类型
        s_churn=0.0,  # 采样器参数
        s_tmin=0.0,  # 采样器参数
        s_tmax=float("inf"),  # 采样器参数
        s_noise=1.0,  # 采样器参数
        steps=40,  # 采样步骤数
        model_path="", # 模型路径
        seed=42, # 随机种子
        ts="",  # 时间步长序列
        in_ch = 4,  # 输入通道数
        out_ch = 3,  # 输出通道数

        dataset_dir = './Coloradar_pre_processing/COLO_RPD_Dataset',  # 数据集目录
        dataloading_config = "./diffusion_consistency_radar/config/data_loading_config_train.yml", # 数据加载配置文件
        output_dir = "",# 输出目录，例如edgar, outdoors, arpg_lab, ec_hallways, aspen, longboard
        gpu_id="0",  # GPU ID
    )
    defaults.update(model_and_diffusion_defaults()) # 更新默认参数，添加模型和扩散相关的默认参数
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    add_dict_to_argparser(parser, defaults) # 将默认参数添加到解析器中
    return parser # 返回解析器对象


if __name__ == "__main__":
    main()
