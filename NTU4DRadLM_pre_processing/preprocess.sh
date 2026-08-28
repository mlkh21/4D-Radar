#!/bin/bash

# ==============================================================================
# NTU4DRadLM 机载高动态数据预处理一键脚本 
# ==============================================================================


# 默认物理参数配置（针对无人机高动态任务量身定制）
DEFAULT_VX=50.0        # legacy fixed 模式参数；默认 none 模式不会使用
DEFAULT_RADAR_LIDAR_MAX_DELTA=0.045 # 12Hz Radar 对 10Hz LiDAR 的正常最近邻窗口
DEFAULT_RADAR_IR_MAX_DELTA=0.025    # 25Hz Thermal 半周期加采样抖动余量
DEFAULT_MAX_REJECTED_FRACTION=0.01  # 只允许跳过少量掉帧型 Radar-LiDAR 候选
VELOCITY_MODE="${VELOCITY_MODE:-none}"
VELOCITY_FRAME="${VELOCITY_FRAME:-radar}"
VELOCITY_FILE="${VELOCITY_FILE:-}"
VELOCITY_MAX_DELTA="${VELOCITY_MAX_DELTA:-0.02}"
RADAR_LIDAR_MAX_DELTA="${RADAR_LIDAR_MAX_DELTA:-$DEFAULT_RADAR_LIDAR_MAX_DELTA}"
RADAR_IR_MAX_DELTA="${RADAR_IR_MAX_DELTA:-$DEFAULT_RADAR_IR_MAX_DELTA}"
MAX_REJECTED_FRACTION="${MAX_REJECTED_FRACTION:-$DEFAULT_MAX_REJECTED_FRACTION}"

# 解析输入的命令行参数
VX=${1:-$DEFAULT_VX}
VISIBILITY_MODE="${VISIBILITY_MODE:-preserve}"

if [[ "$VISIBILITY_MODE" != "preserve" && "$VISIBILITY_MODE" != "hard" ]]; then
    echo "Error: VISIBILITY_MODE must be preserve or hard, got: $VISIBILITY_MODE"
    exit 2
fi
if [[ "$VELOCITY_MODE" != "none" && "$VELOCITY_MODE" != "fixed" && "$VELOCITY_MODE" != "recorded" ]]; then
    echo "Error: VELOCITY_MODE must be none, fixed, or recorded, got: $VELOCITY_MODE"
    exit 2
fi
if [[ "$VELOCITY_FRAME" != "radar" && "$VELOCITY_FRAME" != "lidar" ]]; then
    echo "Error: VELOCITY_FRAME must be radar or lidar, got: $VELOCITY_FRAME"
    exit 2
fi
if [[ "$VELOCITY_MODE" == "recorded" && -z "$VELOCITY_FILE" ]]; then
    echo "Error: VELOCITY_FILE is required when VELOCITY_MODE=recorded"
    exit 2
fi

echo "======================================================================"
echo "🚀 启动 NTU4DRadLM 高动态机载数据预处理流水线"
echo "   固定速度参数  --vx      : ${VX} m/s（仅 fixed 模式生效）"
echo "   时序补偿：读取逐帧 signed delta，仅移动非参考传感器"
echo "   Radar-LiDAR 最大时差   : ${RADAR_LIDAR_MAX_DELTA} s"
echo "   Radar-IR 最大时差      : ${RADAR_IR_MAX_DELTA} s"
echo "   Radar-LiDAR 最大拒绝比例: ${MAX_REJECTED_FRACTION}"
echo "   监督可见性模式           : ${VISIBILITY_MODE}"
echo "   运动补偿模式             : ${VELOCITY_MODE} (${VELOCITY_FRAME})"
echo "======================================================================"

# 获取脚本所在目录的绝对路径，确保在任何路径下执行都能正确定位
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"

# 切换到项目根目录执行，防止相对路径崩溃
cd "$ROOT_DIR" || exit 1

# ------------------------------------------------------------------------------
# STEP 1: 生成时空硬同步映射索引 (雷达帧与激光雷达帧最近邻匹配)
# ------------------------------------------------------------------------------
echo -e "\n[STEP 1/2] 正在计算传感器时空对齐索引 (Timestamp Indexing)..."

if [ -f "$SCRIPT_DIR/NTU4DRadLM_timestamp_index.py" ]; then
    python3 "$SCRIPT_DIR/NTU4DRadLM_timestamp_index.py" \
        --radar_lidar_max_delta "$RADAR_LIDAR_MAX_DELTA" \
        --skip_unmatched \
        --max_rejected_fraction "$MAX_REJECTED_FRACTION"
    
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 时间戳索引生成失败，请检查 Raw 数据集路径是否完整！"
        exit 1
    fi
    echo "✅ STEP 1 成功: 帧对齐文本映射表 (radar_index_sequence.txt) 已完成。"
else
    echo "❌ 错误: 未找到 $SCRIPT_DIR/NTU4DRadLM_timestamp_index.py 文件！"
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 2: 执行机载运动多普勒补偿、多模态红外图缝合与 Voxel 空间切片
# ------------------------------------------------------------------------------
echo -e "\n[STEP 2/2] 正在执行机载高速自身运动多普勒补偿与红外热成像对齐清洗 (Pre-processing Matrix)..."

if [ -f "$SCRIPT_DIR/NTU4DRadLM_pre_processing.py" ]; then
    # 💡 严格核对：确保所有变量与路径的双引号完美闭合，消除所有 Bad Token
    python3 "$SCRIPT_DIR/NTU4DRadLM_pre_processing.py" \
        --vx "$VX" \
        --velocity_mode "$VELOCITY_MODE" --velocity_frame "$VELOCITY_FRAME" \
        --velocity_file "$VELOCITY_FILE" --velocity_max_delta "$VELOCITY_MAX_DELTA" \
        --radar_lidar_max_delta "$RADAR_LIDAR_MAX_DELTA" \
        --radar_ir_max_delta "$RADAR_IR_MAX_DELTA" \
        --visibility_mode "$VISIBILITY_MODE"
    
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 预处理核心矩阵解算失败！"
        exit 1
    fi
    echo "✅ STEP 2 成功: 抗畸变稠密多模态体素切片及 _ir.npy 阵列构建完成。"
else
    echo "❌ 错误: 未找到 $SCRIPT_DIR/NTU4DRadLM_pre_processing.py 文件！"
    exit 1
fi

echo -e "\n======================================================================"
echo "🎉 所有解包后预处理工序圆满完成！数据已转换为可直接输入训练网络的体素格式。"
echo "💡 下一步提示: 运行 bash diffusion_consistency_radar/launch/train_unified.sh 启动训练"
echo "======================================================================"
