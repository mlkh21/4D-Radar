#!/bin/bash
# Fast minimal training for quick validation inside test/mini-test.
# Usage:
#   bash test/mini-test/train_minimal.sh vae
#   bash test/mini-test/train_minimal.sh ldm
#   bash test/mini-test/train_minimal.sh all

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/diffusion_consistency_radar"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
DEFAULT_CONFIG_PATH="${PROJECT_DIR}/config/default_config.yaml"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"

PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre"
MINI_DATASET_DIR="${MINI_DATASET_DIR:-${SELF_DIR}/.tmp_mini_train_dataset}"
MINI_CONFIG_PATH="${MINI_CONFIG_PATH:-${SELF_DIR}/.default_config.mini_override.yaml}"
MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-${SELF_DIR}/train_results_mini}"

MODE="${1:-all}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-200}"

MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-2}"
MINI_GRAD_ACCUM="${MINI_GRAD_ACCUM:-1}"
MINI_USE_AUG="${MINI_USE_AUG:-false}"

MINI_VAE_EPOCHS="${MINI_VAE_EPOCHS:-3}"
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-2}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON_CMD=("${PYTHON_BIN}")
elif python -c "import torch" >/dev/null 2>&1; then
	PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
	PYTHON_CMD=(conda run -n Radar-Diffusion python)
else
	PYTHON_CMD=(python)
fi

if [[ ! -f "${DATA_LOADING_CONFIG}" ]]; then
	echo "Error: data loading config not found: ${DATA_LOADING_CONFIG}"
	exit 1
fi

mapfile -t TRAIN_SCENES < <("${PYTHON_CMD[@]}" - "${DATA_LOADING_CONFIG}" <<'PY'
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

if [[ ${#TRAIN_SCENES[@]} -eq 0 ]]; then
	echo "Error: data.train is empty in ${DATA_LOADING_CONFIG}"
	exit 1
fi

echo "=========================================="
echo "Minimal training setup"
echo "mode: ${MODE}"
echo "train scenes: ${TRAIN_SCENES[*]}"
echo "samples per scene: ${SAMPLES_PER_SCENE}"
echo "project dir: ${PROJECT_DIR}"
echo "results dir: ${MINI_RESULTS_DIR}"
echo "mini dataset dir: ${MINI_DATASET_DIR}"
echo "=========================================="

rm -rf "${MINI_DATASET_DIR}"
mkdir -p "${MINI_DATASET_DIR}"

for SCENE in "${TRAIN_SCENES[@]}"; do
	SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
	SRC_RADAR_DIR="${SRC_SCENE_DIR}/radar_voxel"
	SRC_TARGET_DIR="${SRC_SCENE_DIR}/target_voxel"
	DST_SCENE_DIR="${MINI_DATASET_DIR}/${SCENE}"
	DST_RADAR_DIR="${DST_SCENE_DIR}/radar_voxel"
	DST_TARGET_DIR="${DST_SCENE_DIR}/target_voxel"

	if [[ ! -d "${SRC_RADAR_DIR}" || ! -d "${SRC_TARGET_DIR}" ]]; then
		echo "Error: missing radar_voxel/target_voxel in ${SRC_SCENE_DIR}"
		exit 1
	fi

	mkdir -p "${DST_RADAR_DIR}" "${DST_TARGET_DIR}"

	mapfile -t RADAR_FILES < <(ls "${SRC_RADAR_DIR}" | grep -E '\.(npy|npz)$' | sort | head -n "${SAMPLES_PER_SCENE}")
	if [[ ${#RADAR_FILES[@]} -eq 0 ]]; then
		echo "Error: no radar files found in ${SRC_RADAR_DIR}"
		exit 1
	fi

	for FILE_NAME in "${RADAR_FILES[@]}"; do
		SRC_RADAR_PATH="${SRC_RADAR_DIR}/${FILE_NAME}"
		SRC_TARGET_PATH="${SRC_TARGET_DIR}/${FILE_NAME}"
		if [[ ! -f "${SRC_TARGET_PATH}" ]]; then
			if [[ "${FILE_NAME}" == *.npy && -f "${SRC_TARGET_DIR}/${FILE_NAME%.npy}.npz" ]]; then
				SRC_TARGET_PATH="${SRC_TARGET_DIR}/${FILE_NAME%.npy}.npz"
			elif [[ "${FILE_NAME}" == *.npz && -f "${SRC_TARGET_DIR}/${FILE_NAME%.npz}.npy" ]]; then
				SRC_TARGET_PATH="${SRC_TARGET_DIR}/${FILE_NAME%.npz}.npy"
			else
				continue
			fi
		fi

		ln -s "${SRC_RADAR_PATH}" "${DST_RADAR_DIR}/$(basename "${SRC_RADAR_PATH}")"
		ln -s "${SRC_TARGET_PATH}" "${DST_TARGET_DIR}/$(basename "${SRC_TARGET_PATH}")"
	done
done

mkdir -p "${MINI_RESULTS_DIR}/vae" "${MINI_RESULTS_DIR}/ldm"

"${PYTHON_CMD[@]}" - "${DEFAULT_CONFIG_PATH}" "${MINI_CONFIG_PATH}" "${MINI_DATASET_DIR}" "${MINI_BATCH_SIZE}" "${MINI_NUM_WORKERS}" "${MINI_USE_AUG}" "${MINI_VAE_EPOCHS}" "${MINI_LDM_EPOCHS}" "${MINI_GRAD_ACCUM}" "${MINI_RESULTS_DIR}" <<'PY'
import sys
import yaml

(
		src_cfg,
		dst_cfg,
		dataset_dir,
		batch_size,
		num_workers,
		use_aug,
		vae_epochs,
		ldm_epochs,
		grad_accum,
		results_dir,
) = sys.argv[1:11]

with open(src_cfg, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f) or {}

cfg.setdefault('data', {})
cfg['data']['dataset_dir'] = dataset_dir
cfg['data']['batch_size'] = int(batch_size)
cfg['data']['num_workers'] = int(num_workers)
cfg['data']['use_augmentation'] = str(use_aug).lower() in {'1', 'true', 'yes', 'on'}

cfg.setdefault('vae', {})
cfg['vae']['epochs'] = int(vae_epochs)
cfg['vae']['save_every'] = 1
cfg['vae']['save_dir'] = f"{results_dir}/vae"

cfg.setdefault('ldm', {})
cfg['ldm']['epochs'] = int(ldm_epochs)
cfg['ldm']['save_every'] = 1
cfg['ldm']['save_dir'] = f"{results_dir}/ldm"

cfg.setdefault('optimization', {})
cfg['optimization']['gradient_accumulation_steps'] = int(grad_accum)

with open(dst_cfg, 'w', encoding='utf-8') as f:
		yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

echo "Mini config: ${MINI_CONFIG_PATH}"
echo "Mini dataset: ${MINI_DATASET_DIR}"

run_vae() {
	CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/unified_train.py" \
		--mode vae \
		--config "${MINI_CONFIG_PATH}"
}

run_ldm() {
	local vae_ckpt="${MINI_RESULTS_DIR}/vae/vae_best.pt"
	if [[ ! -f "${vae_ckpt}" ]]; then
		echo "Error: minimal VAE checkpoint not found: ${vae_ckpt}"
		echo "Run VAE first: bash ${SELF_DIR}/train_minimal.sh vae"
		exit 1
	fi

	CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/unified_train.py" \
		--mode ldm \
		--config "${MINI_CONFIG_PATH}" \
		--vae_ckpt "${vae_ckpt}"
}

case "${MODE}" in
	vae)
		run_vae
		;;
	ldm)
		run_ldm
		;;
	all)
		run_vae
		run_ldm
		;;
	*)
		echo "Usage: $0 [vae|ldm|all]"
		exit 1
		;;
esac

echo "Minimal training done."
