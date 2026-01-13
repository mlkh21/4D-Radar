# -- coding: utf-8 --
# 训练一个扩散模型来生成雷达数据。

import argparse
import sys
import os

# 强制将当前项目的路径添加到 sys.path 的最前面，以优先加载当前项目的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm import dist_util, logger
from mpi4py import MPI
# from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util_cond import TrainLoop
import torch.distributed as dist
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

import torch as th
import os 

from fvcore.nn import parameter_count_table, FlopCountAnalysis
import yaml
from easydict import EasyDict as edict

# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 移到 main 中根据参数设置

def main():
    parser = create_argparser()
    # 添加配置文件参数
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids[rank % len(gpu_ids)]

    dist_util.setup_dist()
    if th.cuda.is_available():
        th.cuda.set_device(0)
    logger.configure(dir = args.out_dir)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("Model created, moving to device...")
    model.to(dist_util.dev())
    print("Model moved to device.")
    print("dist_util.dev()", dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # 开启 cuDNN benchmark 以加速固定尺寸输入的训练
    if th.cuda.is_available():
        th.backends.cudnn.benchmark = True

    print(parameter_count_table(model))

    logger.log("creating data loader...")
    batch_size = args.batch_size
    
    # 数据集配置路径 = args 数据加载配置
    # with open(数据集配置路径, 'r') 作为 fid:
    # coloradar config = edict(yaml load(fid, Loader=yaml FullLoader))

    # tran list = [转换 Lambda(lambda x: torch from numpy(x)), 转换 Resize((args 图像大小,args 图像大小))]
    # 变换训练 = 变换 Compose(tran list)
    # 训练数据 = 初始化数据集 voxel(coloradar config, args 数据集目录, 变换训练, "训练")

    from cm.dataset_loader import NTU4DRadLM_VoxelDataset
    train_data = NTU4DRadLM_VoxelDataset(
        root_dir=args.dataset_dir,
        split='train',
    )

    print("batch_size", batch_size)
    data= th.utils.data.DataLoader(
        train_data,
        num_workers=args.num_workers, # 使用 Args 中的 num_workers
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True, # 开启锁页内存，加速数据从 CPU 到 GPU 的传输
        persistent_workers=True, # 保持 worker 进程存活，减少重建开销
        drop_last=True
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        schedule_sampler="lognormal", # 采样时间步的调度器。影响训练过程中噪声水平的采样分布。
        lr=0.0001, # 学习率 (优化: 0.00005 -> 0.0001)
        weight_decay=0.0, # 权重衰减（L2 正则化）
        lr_anneal_steps=0, # 学习率退火步数
        global_batch_size=32, # 全局批量大小
        batch_size=4, # 单个 GPU 的批量大小
        microbatch=-1, # 微批量大小（梯度累积）
        ema_rate="0.999,0.9999,0.9999432189950708", # EMA 衰减率
        log_interval=100, # 日志打印间隔
        save_interval=40000, # 模型保存间隔
        resume_checkpoint="", # 恢复训练的 checkpoint 路径
        use_fp16=True, # 混合精度训练 (优化: False -> True)
        fp16_scale_growth=1e-3, # FP16 Loss Scale 增长速率
        in_ch=8, # 模型输入通道数
        out_ch=4, # 模型输出通道数
        out_dir='./diffusion_consistency_radar/train_results/radar_edm', # 输出目录
        dataset_dir='./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre', # 数据集目录
        dataloading_config="./diffusion_consistency_radar/config/data_loading_config.yml", # 数据配置
        num_workers=4, # 数据加载线程数 (优化: 2 -> 4)

        # === 模型架构参数 (优化后的默认值) ===
        attention_resolutions="8,4", # 注意力分辨率 (优化: "32,16,8" -> "8,4")
        class_cond=False, # 类别条件
        use_scale_shift_norm=True, # Scale-Shift Normalization
        dropout=0.1, # Dropout 率
        image_size=128, # 输入分辨率
        num_channels=64, # 基础通道数 (优化: 128 -> 64)
        num_head_channels=64, # 每头通道数
        num_res_blocks=2, # 残差块数
        resblock_updown=True, # 残差块上下采样 (优化: False -> True)
        weight_schedule="karras", # 权重调度
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        
        # === 新增优化参数 ===
        attention_type="flash", # 注意力类型
        norm_type="group", # 归一化类型
        downsample_type="asymmetric", # 下采样类型
        downsample_stride="xy_only", # 下采样步长
        use_optimized_unet=True, # 使用优化版 UNet
        use_depthwise=False, # 深度可分离卷积
        window_size="4,4,4", # 窗口大小
        initial_z_size=32, # 初始 Z 轴大小
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
