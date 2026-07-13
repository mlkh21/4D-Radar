#!/bin/bash
# 启动 v9A 顶部过冲权重实验，具体协议由通用 screen 脚本统一维护。

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V9_VARIANT="A"
exec "${SELF_DIR}/run_ldm_z64_v9_screen.sh" "$@"

