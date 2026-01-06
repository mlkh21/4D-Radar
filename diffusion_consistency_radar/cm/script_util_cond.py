import argparse # 导入参数解析库，用于处理命令行参数

from .karras_diffusion import KarrasDenoiser # 导入 Karras 扩散过程类（核心算法）
from .unet import UNetModel # 导入 U-Net 模型类（神经网络架构）
import numpy as np # 导入 NumPy 库

NUM_CLASSES = 1000 # 定义默认类别数（ImageNet 标准），但在雷达任务中可能未被实际使用


def cm_train_defaults():
    """
    输入:
        无
    输出:
        (dict) - 包含一致性模型训练默认参数的字典
    作用: 提供一致性模型 (Consistency Model) 训练所需的默认超参数。
    逻辑:
    1. 返回一个包含 teacher_model_path, training_mode, target_ema_mode 等键值对的字典。
    """
    return dict(
        teacher_model_path="", # 教师模型路径（用于蒸馏），默认为空
        teacher_dropout=0.1, # 教师模型的 Dropout 率
        training_mode="consistency_distillation", # 训练模式：默认为一致性蒸馏
        target_ema_mode="fixed", # 目标模型 EMA 模式：固定
        scale_mode="fixed", # 缩放模式：固定
        total_training_steps=600000, # 总训练步数
        start_ema=0.0, # 起始 EMA 衰减率
        start_scales=40, # 起始缩放级数（离散化步数）
        end_scales=40, # 结束缩放级数
        distill_steps_per_iter=50000, # 每次迭代的蒸馏步数
        loss_norm="lpips", # 损失函数使用的范数类型（如 LPIPS 感知损失）
    )


def model_and_diffusion_defaults():
    """
    输入:
        无
    输出:
        (dict) - 包含模型和扩散过程默认参数的字典
    作用: 提供 U-Net 模型和扩散过程所需的默认超参数。
    逻辑:
    1. 返回一个包含 image_size, num_channels, num_res_blocks, in_ch, out_ch 等键值对的字典。
    """
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002, # 最小噪声标准差
        sigma_max=80.0, # 最大噪声标准差
        image_size=64, # 图像尺寸（默认64，但在训练脚本中会被覆盖为128）
        num_channels=128, # U-Net 基础通道数
        num_res_blocks=2, # 每个分辨率层级的残差块数量
        num_heads=4, # 注意力机制的头数
        in_ch = 3, # 输入通道数（默认3，但在雷达任务中会被改为2）
        out_ch = 6, # 输出通道数（默认6，但在雷达任务中会被改为1）
        num_heads_upsample=-1, # 上采样层的注意力头数（-1表示不使用）
        num_head_channels=-1, # 每个头的通道数（-1表示自动计算）
        attention_resolutions="32,16,8", # 在哪些分辨率层级使用注意力机制
        channel_mult="", # 通道倍增系数（控制每层通道数的变化）
        dropout=0.0, # Dropout 率
        class_cond=False, # 是否使用类别条件（Class Conditioning）
        use_checkpoint=False, # 是否使用梯度检查点（节省显存）
        use_scale_shift_norm=True, # 是否使用 Scale-Shift Normalization
        resblock_updown=False, # 是否在残差块内进行上下采样
        use_fp16=False, # 是否使用混合精度训练
        use_new_attention_order=False, # 是否使用新的注意力顺序
        learn_sigma=False, # 是否学习方差（通常用于改进的 DDPM）
        weight_schedule="karras", # 权重调度策略（Karras 论文推荐）
        dims=3, # 维度 (2 for 2D, 3 for 3D)
    )
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    in_ch,
    out_ch,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    weight_schedule,
    dims=3,
    sigma_min=0.002,
    sigma_max=80.0,
    distillation=False,
):
    """
    输入:
        image_size (int) - 图像尺寸
        class_cond (bool) - 是否使用类别条件
        learn_sigma (bool) - 是否学习方差
        num_channels (int) - 基础通道数
        num_res_blocks (int) - 残差块数量
        channel_mult (str/tuple) - 通道倍增系数
        in_ch (int) - 输入通道数
        out_ch (int) - 输出通道数
        num_heads (int) - 注意力头数
        num_head_channels (int) - 每个头的通道数
        num_heads_upsample (int) - 上采样注意力头数
        attention_resolutions (str) - 注意力分辨率
        dropout (float) - Dropout 率
        use_checkpoint (bool) - 是否使用梯度检查点
        use_scale_shift_norm (bool) - 是否使用 Scale-Shift Normalization
        resblock_updown (bool) - 是否在残差块内上下采样
        use_fp16 (bool) - 是否使用 FP16
        use_new_attention_order (bool) - 是否使用新注意力顺序
        weight_schedule (str) - 权重调度策略
        dims (int) - 维度 (2 或 3)
        sigma_min (float) - 最小噪声标准差
        sigma_max (float) - 最大噪声标准差
        distillation (bool) - 是否蒸馏模式
    输出:
        model (UNetModel) - 创建的 U-Net 模型
        diffusion (KarrasDenoiser) - 创建的扩散过程对象
    作用: 创建并初始化 U-Net 模型和 Karras 扩散器。
    逻辑:
    1. 调用 create_model 创建 U-Net 模型。
    2. 初始化 KarrasDenoiser 对象。
    3. 返回模型和扩散器。
    """
    # print("in_ch", in_ch)
    # print("out_ch", out_ch) # 打印输出通道数（调试用）
    # print("dims", dims)
    model = create_model( # 调用 create_model 函数创建 U-Net 模型
        image_size,
        num_channels,
        num_res_blocks,
        in_ch = in_ch, # 传递输入通道数
        out_ch = out_ch, # 传递输出通道数
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        dims=dims, # 传递维度
    )
    diffusion = KarrasDenoiser( # 创建 KarrasDenoiser 对象（负责扩散过程的数学计算）
        sigma_data=0.5, # 数据分布的标准差假设
        sigma_max=sigma_max, # 最大噪声
        sigma_min=sigma_min, # 最小噪声
        distillation=distillation, # 是否蒸馏模式
        weight_schedule=weight_schedule, # 权重调度
        loss_norm="lpips",
    )
    return model, diffusion # 返回模型和扩散对象


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    in_ch=3,
    out_ch=6,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    dims=3,
):
    """
    输入:
        (同 create_model_and_diffusion 中的对应参数)
    输出:
        (UNetModel) - 实例化的 U-Net 模型
    作用: 根据配置参数构建 U-Net 模型。
    逻辑:
    1. 解析 channel_mult 参数，如果为空则根据 image_size 设置默认值。
    2. 解析 attention_resolutions 参数。
    3. 实例化并返回 UNetModel。
    """
    if channel_mult == "":
        if dims == 3 and image_size == 128:
             channel_mult = (1, 2, 4, 8) # 3D U-Net default for 128x128xZ
        elif image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128: # 你的项目用的是 128
            channel_mult = (1, 1, 2, 3, 4) # 通道数变化：1x -> 1x -> 2x -> 3x -> 4x
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(",")) # 解析字符串格式的倍增系数

    attention_ds = [] # 解析注意力分辨率
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res)) # 将分辨率转换为下采样倍数

    return UNetModel( # 实例化 UNetModel
        image_size=image_size,
        in_channels = in_ch, # 设置输入通道（雷达任务中为2）
        model_channels=num_channels,
        # out_channels=(3 if not learn_sigma else 6), # 原版代码（注释掉）
        out_channels = out_ch, # 设置输出通道（雷达任务中为1）
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=dims, # Pass dims
    )


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    """
    输入:
        target_ema_mode (str) - 目标模型 EMA 模式 ('fixed', 'adaptive')
        start_ema (float) - 起始 EMA 率
        scale_mode (str) - 缩放模式 ('fixed', 'progressive', 'progdist')
        start_scales (int) - 起始缩放级数
        end_scales (int) - 结束缩放级数
        total_steps (int) - 总训练步数
        distill_steps_per_iter (int) - 每次迭代的蒸馏步数
    输出:
        ema_and_scales_fn (function) - 一个接收 step 参数并返回 (target_ema, scales) 的函数
    作用: 创建一个用于计算当前训练步数下的 EMA 率和采样级数 (Scales) 的函数。
    逻辑:
    1. 定义内部函数 ema_and_scales_fn(step)。
    2. 根据 target_ema_mode 和 scale_mode 计算 target_ema 和 scales。
    3. 返回内部函数。
    """
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed": # 固定模式（最简单）
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive": # 渐进式缩放
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive": # 自适应 EMA + 渐进式缩放
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist": # 渐进式蒸馏 (Progressive Distillation)
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales) # 返回计算出的 EMA 率和 Scales 数

    return ema_and_scales_fn # 返回闭包函数


def add_dict_to_argparser(parser, default_dict):
    """
    输入:
        parser (argparse.ArgumentParser) - 参数解析器对象
        default_dict (dict) - 包含默认参数的字典
    输出:
        无
    作用: 将字典中的键值对作为参数添加到 argparse 解析器中。
    逻辑:
    1. 遍历字典中的每个键值对。
    2. 根据值的类型推断参数类型 (bool 类型特殊处理)。
    3. 调用 parser.add_argument 添加参数。
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool # 特殊处理布尔值
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    输入:
        args (argparse.Namespace) - 解析后的参数对象
        keys (iterable) - 需要提取的参数名列表
    输出:
        (dict) - 提取出的参数字典
    作用: 从 argparse 解析结果中提取指定键的参数，组成字典。
    逻辑:
    1. 遍历 keys 列表。
    2. 从 args 中获取对应属性的值。
    3. 构建并返回字典。
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    输入:
        v (str/bool) - 输入值
    输出:
        (bool) - 转换后的布尔值
    作用: 将字符串转换为布尔值。
    逻辑:
    1. 如果输入已经是 bool，直接返回。
    2. 检查字符串是否在真值列表 ('yes', 'true', 't', 'y', '1') 中。
    3. 检查字符串是否在假值列表 ('no', 'false', 'f', 'n', '0') 中。
    4. 否则抛出异常。
    """
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")