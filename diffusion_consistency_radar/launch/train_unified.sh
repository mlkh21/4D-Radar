#!/bin/bash
# ==============================================================================
# 统一训练脚本 - VAE / LDM / CD 一站式训练
# ==============================================================================
#
# 使用方法:
#   sh diffusion_consistency_radar/launch/train_unified.sh vae      # 训练 VAE
#   sh diffusion_consistency_radar/launch/train_unified.sh ldm      # 训练 LDM
#   sh diffusion_consistency_radar/launch/train_unified.sh cd       # 蒸馏 CD
#   sh diffusion_consistency_radar/launch/train_unified.sh all      # 完整流程
#
# 配置文件: config/default_config.yaml
#
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 默认路径
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SELF_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
DEFAULT_CONFIG_PATH="${PROJECT_DIR}/config/default_config.yaml"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"
PREPROCESSED_ROOT="${ROOT_DIR}/NTU4DRadLM_pre_processing/NTU4DRadLM_Pre"
TRAIN_DATASET_DIR="${PROJECT_DIR}/.tmp_train_dataset"
CONFIG_PATH="${PROJECT_DIR}/config/.default_config.train_override.yaml"
RESULTS_DIR="${PROJECT_DIR}/train_results"

if [ ! -f "${DATA_LOADING_CONFIG}" ]; then
    echo "Error: data loading config not found: ${DATA_LOADING_CONFIG}"
    exit 1
fi

mapfile -t TRAIN_SCENES < <(python - "${DATA_LOADING_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

scenes = (cfg.get('data') or {}).get('train') or []
if isinstance(scenes, str):
    scenes = [scenes]

for scene in scenes:
    s = str(scene).strip()
    if s:
        print(s)
PY
)

if [ ${#TRAIN_SCENES[@]} -eq 0 ]; then
    echo "Error: data_loading_config.yml data.train is empty"
    exit 1
fi

rm -rf "${TRAIN_DATASET_DIR}"
mkdir -p "${TRAIN_DATASET_DIR}"

for SCENE in "${TRAIN_SCENES[@]}"; do
    SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
    if [ ! -d "${SRC_SCENE_DIR}" ]; then
        echo "Error: train scene directory not found: ${SRC_SCENE_DIR}"
        exit 1
    fi
    ln -s "${SRC_SCENE_DIR}" "${TRAIN_DATASET_DIR}/${SCENE}"
done

python - "${DEFAULT_CONFIG_PATH}" "${CONFIG_PATH}" "${TRAIN_DATASET_DIR}" <<'PY'
import sys
import yaml

src_cfg, dst_cfg, dataset_dir = sys.argv[1:4]
with open(src_cfg, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('data', {})
cfg['data']['dataset_dir'] = dataset_dir

with open(dst_cfg, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

echo "Using train scenes: ${TRAIN_SCENES[*]}"
echo "Training dataset root: ${TRAIN_DATASET_DIR}"
echo "Training config: ${CONFIG_PATH}"

MODE=${1:-vae}

case "$MODE" in
    vae)
        echo "=========================================="
        echo "Stage 1: Training VAE"
        echo "=========================================="
        
        CUDA_VISIBLE_DEVICES=0,1 python ${SCRIPT_DIR}/unified_train.py \
            --mode vae \
            --config ${CONFIG_PATH}
        ;;
        
    ldm)
        echo "=========================================="
        echo "Stage 2: Training LDM"
        echo "=========================================="
        
        VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
        if [ ! -f "$VAE_CKPT" ]; then
            echo "Error: VAE checkpoint not found at $VAE_CKPT"
            echo "Please train VAE first: sh $0 vae"
            exit 1
        fi
        
        # 检查是否存在 LDM 检查点以便断点续训
        LDM_RESUME="${RESULTS_DIR}/ldm/ldm_best.pt"
        if [ -f "$LDM_RESUME" ]; then
            echo "Found existing LDM checkpoint, resuming from: $LDM_RESUME"
            CUDA_VISIBLE_DEVICES=0,1 python ${SCRIPT_DIR}/unified_train.py \
                --mode ldm \
                --config ${CONFIG_PATH} \
                --vae_ckpt ${VAE_CKPT} \
                --resume ${LDM_RESUME}
        else
            echo "Starting LDM training from scratch"
            CUDA_VISIBLE_DEVICES=0,1 python ${SCRIPT_DIR}/unified_train.py \
                --mode ldm \
                --config ${CONFIG_PATH} \
                --vae_ckpt ${VAE_CKPT}
        fi
        ;;
        
    cd)
        echo "=========================================="
        echo "Stage 3: Consistency Distillation"
        echo "=========================================="
        
        VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
        LDM_CKPT="${RESULTS_DIR}/ldm/ldm_best.pt"
        
        if [ ! -f "$VAE_CKPT" ]; then
            echo "Error: VAE checkpoint not found at $VAE_CKPT"
            exit 1
        fi
        
        if [ ! -f "$LDM_CKPT" ]; then
            echo "Error: LDM checkpoint not found at $LDM_CKPT"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/cd_train_optimized.py \
            --ldm_ckpt ${LDM_CKPT} \
            --vae_ckpt ${VAE_CKPT} \
            --dataset_dir ${TRAIN_DATASET_DIR}
        ;;
        
    all)
        echo "=========================================="
        echo "Running Full Training Pipeline"
        echo "=========================================="
        
        # Stage 1: VAE
        sh $0 vae
        if [ $? -ne 0 ]; then exit 1; fi
        
        # Stage 2: LDM
        sh $0 ldm
        if [ $? -ne 0 ]; then exit 1; fi
        
        # Stage 3: CD
        sh $0 cd
        ;;
        
    *)
        echo "Usage: $0 [vae|ldm|cd|all]"
        echo ""
        echo "  vae  - Train VAE (Stage 1)"
        echo "  ldm  - Train Latent Diffusion (Stage 2)"
        echo "  cd   - Consistency Distillation (Stage 3)"
        echo "  all  - Run full pipeline"
        exit 1
        ;;
esac

echo "Done!"
