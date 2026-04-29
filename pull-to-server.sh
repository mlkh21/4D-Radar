#!/bin/bash

# ================= 配置区域 =================
# 远程服务器信息
DEST_USER="ps"
DEST_IP="100.103.244.70"
DEST_PASS="1"  # 在这里输入你的实际密码
PORT="22"

# 路径信息 (注意：源路径末尾不带斜杠，会同步整个文件夹)
SOURCE="/home/zxj/catkin_ws/src/4D-Radar-Diffusion/"
DEST_PATH="/home/ps/zxj_workspace/src/4D-Radar/"

# 传输设置
BW_LIMIT="50000"  # 限速 50MB/s
RETRY_DELAY=5     # 断线重试间隔（秒）
# ============================================

echo "开始数据传输任务..."

# 使用 until 循环：直到 rsync 命令返回成功（状态码 0），否则一直循环
until sshpass -p "$DEST_PASS" rsync -avP \
    --append-verify \
    --bwlimit=$BW_LIMIT \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='Data/' \
    --exclude='Result/' \
    -e "ssh -p $PORT -o ConnectTimeout=10 -o ServerAliveInterval=30" \
    "$SOURCE" \
    "$DEST_USER@$DEST_IP:$DEST_PATH"; do
    
    EXIT_CODE=$?
    echo "------------------------------------------------"
    echo "警告：传输异常中断（错误码：$EXIT_CODE）。"
    echo "$RETRY_DELAY 秒后将自动尝试断点续传..."
    echo "------------------------------------------------"
    sleep $RETRY_DELAY
done

echo "================================================"
echo "恭喜！所有数据已成功同步并校验完成。"
echo "================================================"
