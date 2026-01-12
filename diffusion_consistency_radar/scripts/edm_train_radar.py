# -*- coding: utf-8 -*-
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

    # 1. (已移除) 加载 YAML 配置

    # 2. 设置环境
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_ids = [x.strip() for x in args.gpu_id.split(',')]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids[rank % len(gpu_ids)]

    dist_util.setup_dist()
    logger.configure(dir = args.out_dir)

    # 3. 使用 Args 代替 Config

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
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
        schedule_sampler="lognormal", # 采样时间步的调度器。影响训练过程中噪声水平的采样分布。可选 "uniform", "lognormal" 等。Lognormal 通常用于 EDM 训练，强调中间噪声水平。
        lr=0.00005, # 学习率。影响模型收敛速度和稳定性。过大可能导致震荡，过小收敛慢。通常在 1e-4 到 1e-5 之间。
        weight_decay=0.0, # 权重衰减（L2 正则化）。防止过拟合。通常设为 0 或很小的值（如 1e-4）。
        lr_anneal_steps=0, # 学习率退火步数。0 表示不退火（常数学习率）。如果设置，学习率会在这些步数内衰减到 0。
        global_batch_size=32, # 全局批量大小（所有 GPU 总和）。针对 2x 4090 调整 (16 * 2 = 32)。
        batch_size=16, # 单个 GPU 的批量大小。针对 RTX 4090 (24GB) 优化，预计占用 ~22GB 显存。
        microbatch=-1,  # 微批量大小。用于梯度累积。-1 表示禁用（即 microbatch = batch_size）。如果显存不足，可以设小一点（如 1 或 2），通过多次前向传播累积梯度来模拟大 batch_size。
        ema_rate="0.999,0.9999,0.9999432189950708",  # 指数移动平均（EMA）的衰减率列表。用于平滑模型参数，通常能获得更好的生成质量。逗号分隔表示维护多个 EMA 版本。值越接近 1，平滑程度越高。
        log_interval=100, # 日志打印间隔（步数）。影响控制台输出频率。
        save_interval=40000, # 模型保存间隔（步数）。影响 checkpoint 文件的生成频率。
        resume_checkpoint="", # 恢复训练的 checkpoint 路径。为空表示从头训练。填入路径可继续训练。
        use_fp16=False, # 是否使用混合精度（FP16）训练。True 可以减少显存占用并加速，但可能导致数值不稳定。
        fp16_scale_growth=1e-3, # FP16 训练时 Loss Scale 的增长速率。仅在 use_fp16=True 时有效。
        in_ch = 8, # 模型输入通道数。
        out_ch = 3, # 模型输出通道数。
        out_dir='./diffusion_consistency_radar/train_results/radar_edm', # 输出目录。存放日志和模型权重。
        dataset_dir = './NTU4DRadLM_pre_processing/NTU4DRadLM_Pre', # 数据集根目录路径。
        dataloading_config = "./diffusion_consistency_radar/config/data_loading_config.yml", # 数据加载配置文件路径。
        num_workers=4, # 数据加载线程数

        # Model params (模型架构参数)
        attention_resolutions="32,16,8", # 在哪些分辨率层级使用自注意力机制。逗号分隔。通常在低分辨率层使用以捕捉全局依赖，高分辨率层使用计算代价过大。
        class_cond=False, # 是否使用类别条件控制（Class Conditioning）。True 则模型会接收类别标签作为输入。
        use_scale_shift_norm=True, # 是否使用 Scale-Shift Normalization（通常用于条件注入，如时间步嵌入）。True 通常效果更好。
        dropout=0.1, # Dropout 概率。防止过拟合。0.1 表示丢弃 10% 的神经元。
        image_size=64, # 输入图像的分辨率（宽/高）。必须与数据预处理一致。
        num_channels=128, # 基础通道数（第一层的宽度）。控制模型容量。越大模型越强，但计算量和显存占用越大。
        num_head_channels=64, # 注意力机制中每个头的通道数。-1 表示使用固定数量的头。
        num_res_blocks=2, # 每个分辨率层级的残差块数量。越多模型越深，容量越大。
        resblock_updown=True, # 是否在上下采样层使用残差块。True 通常能保留更多信息。
        weight_schedule="karras", # 权重调度策略。影响不同噪声水平下 Loss 的权重分配。Karras 策略通常用于 EDM，旨在平衡不同噪声等级的学习。
        sigma_min=0.002, 
        sigma_max=80.0,
        rho=7.0,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
