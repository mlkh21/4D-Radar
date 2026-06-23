#!/bin/bash

# ==============================================================================
# NTU4DRadLM 机载高动态数据预处理一键脚本 
# ==============================================================================


# 默认物理参数配置（针对无人机高动态任务量身定制）
DEFAULT_VX=50.0        # 默认无人机前进速度 (m/s)，对应 180 km/h
DEFAULT_DT=0.002       # 默认红外-雷达硬件触发时钟残差 (秒)，即 2ms

# 解析输入的命令行参数
VX=${1:-$DEFAULT_VX}
DT_SYNC=${2:-$DEFAULT_DT}

echo "======================================================================"
echo "🚀 启动 NTU4DRadLM 高动态机载数据预处理流水线"
echo "   设定巡航速度  --vx      : ${VX} m/s"
echo "   设定硬件时滞  --dt_sync : ${DT_SYNC} s"
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
    python3 "$SCRIPT_DIR/NTU4DRadLM_timestamp_index.py"
    
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
    python3 "$SCRIPT_DIR/NTU4DRadLM_pre_processing.py" --vx "$VX" --dt_sync "$DT_SYNC" --require_radar_visibility
    
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