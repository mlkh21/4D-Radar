"""
Train a diffusion model on images.
"""

import argparse
import sys
import os

# 强制将当前项目的路径添加到 sys.path 的最前面，以优先加载当前项目的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm import dist_util, logger
# from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util_cond import CMTrainLoop
from torchvision import transforms
import torch.distributed as dist

import torch as th
import yaml
from easydict import EasyDict as edict
import copy
import os 

from fvcore.nn import parameter_count_table
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir = args.out_dir)

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    print("ema_scale_fn", ema_scale_fn)
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size


    
    dataset_config_path = args.dataloading_config 
    # with open(dataset_config_path, 'r') as fid:
    #     coloradar_config = edict(yaml.load(fid, Loader=yaml.FullLoader))
    
    # tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    # transform_train = transforms.Compose(tran_list)
    # train_data = init_dataset(coloradar_config, args.dataset_dir, transform_train, "train")
    
    # Load Dataset using our new VoxelDataset
    from cm.dataset_loader import NTU4DRadLM_VoxelDataset
    train_data = NTU4DRadLM_VoxelDataset(
        root_dir=args.dataset_dir,
        split='train'
    )

    print("batch_size", batch_size)
    data= th.utils.data.DataLoader(
        train_data,
        num_workers=4, # 减少 worker 数量以避免 OOM
        batch_size=batch_size,
        shuffle=True,
        drop_last=True) # 添加 drop_last
    print("args.teacher_model_path", args.teacher_model_path)
    if len(args.teacher_model_path) > 0:  # 教师评分模型的路径
        # print("loading the teacher model from")
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,
    )

    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    logger.log("training...")
    # print(parameter_count_table(model))
    print(parameter_count_table(teacher_model))
    print(parameter_count_table(target_model))
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
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
    defaults = dict(
        # --- 在这里修改默认值 ---
        teacher_model_path = "", # 预设教师模型路径 (需填入预训练好的EDM模型)
        schedule_sampler="uniform",
        lr=0.00005, # 修改学习率
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=4, # 预设 Batch Size (注意这里改的是 global_batch_size)
        batch_size=-1,
        microbatch=-1, 
        ema_rate="0.999,0.9999,0.9999432189950708", # 预设 EMA Rate
        log_interval=100,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False, # 预设关闭 FP16
        fp16_scale_growth=1e-3,
        in_ch = 4, # 预设输入通道 4 (Radar Voxel)
        out_ch = 4, # 预设输出通道 4 (Target Voxel)
        out_dir='./diffusion_consistency_radar/train_results/radar_cd', # 预设输出目录
        dataset_dir = '/home/zxj/catkin_ws/src/4D-Radar-Diffusion/NTU4DRadLM_pre_processing/NTU4DRadLM_Pre', # 指向预处理后的数据
        dataloading_config = "./diffusion_consistency_radar/config/data_loading_config_train.yml", # 确保指向正确的配置文件
        
        # --- 其他参数 (来自 cm_train_defaults 和 model_and_diffusion_defaults) ---
        # 你也可以在这里覆盖其他默认值，例如：
        training_mode="consistency_distillation",
        target_ema_mode="fixed",
        start_ema=0.95,
        scale_mode="fixed",
        start_scales=40,
        loss_norm="l2",
        attention_resolutions="32,16,8",
        class_cond=False,
        use_scale_shift_norm=True,
        dropout=0.0,
        teacher_dropout=0.1,
        image_size=128,
        num_channels=64,
        num_head_channels=64,
        num_res_blocks=3,
        resblock_updown=True,
        weight_schedule="uniform",
    )
    
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
