#!/bin/bash
# Fast minimal training for quick validation inside test/mini-test.
# Usage:
#   bash test/mini-test/train_minimal.sh vae
#   bash test/mini-test/train_minimal.sh ldm
#   bash test/mini-test/train_minimal.sh cd
#   bash test/mini-test/train_minimal.sh all
#   bash test/mini-test/train_minimal.sh all_with_cd

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SELF_DIR}/../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/diffusion_consistency_radar"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
DEFAULT_CONFIG_PATH="${PROJECT_DIR}/config/default_config.yaml"
DATA_LOADING_CONFIG="${PROJECT_DIR}/config/data_loading_config.yml"

MINI_RADAR_PROTOCOL="${MINI_RADAR_PROTOCOL:-legacy}"
case "${MINI_RADAR_PROTOCOL}" in
	legacy)
		DEFAULT_PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware"
		DEFAULT_MINI_TARGET_SIZE="32,128,128"
		DEFAULT_MINI_SOURCE_PC_RANGE="0,-20,-6,120,20,10"
		DEFAULT_MINI_MODEL_PC_RANGE="0,-20,-6,40,20,10"
		DEFAULT_MINI_DATASET_DIR="${SELF_DIR}/.tmp_mini_train_dataset"
		;;
	formal)
		DEFAULT_PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
		DEFAULT_MINI_TARGET_SIZE="32,128,128"
		DEFAULT_MINI_SOURCE_PC_RANGE="0,-20,-6,80,20,10"
		DEFAULT_MINI_MODEL_PC_RANGE="0,-20,-6,80,20,10"
		DEFAULT_MINI_DATASET_DIR="${DEFAULT_PREPROCESSED_ROOT}"
		;;
	*)
		echo "Error: MINI_RADAR_PROTOCOL must be legacy or formal"
		exit 1
		;;
esac
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${DEFAULT_PREPROCESSED_ROOT}}"
CALIB_CONFIG_DIR="${CALIB_CONFIG_DIR:-${ROOT_DIR}/Data/config}"
if [[ ${MINI_DATASET_DIR+x} == x && -z "${MINI_DATASET_DIR}" ]]; then
	echo "Error: unsafe MINI_DATASET_DIR: path is empty"
	exit 1
fi
MINI_DATASET_DIR="${MINI_DATASET_DIR:-${DEFAULT_MINI_DATASET_DIR}}"
MINI_DATASET_DIR_INPUT="${MINI_DATASET_DIR}"
MINI_CONFIG_PATH="${MINI_CONFIG_PATH:-${SELF_DIR}/.default_config.mini_override.yaml}"
MINI_CONFIG_PATH_INPUT="${MINI_CONFIG_PATH}"
MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-${SELF_DIR}/train_results_mini}"
MINI_REQUIRE_FRESH_SCRATCH="${MINI_REQUIRE_FRESH_SCRATCH:-0}"
MINI_REQUIRE_FRESH_CONFIG="${MINI_REQUIRE_FRESH_CONFIG:-0}"
MINI_PREFLIGHT_ONLY="${MINI_PREFLIGHT_ONLY:-0}"
MINI_TARGET_SIZE="${MINI_TARGET_SIZE:-${DEFAULT_MINI_TARGET_SIZE}}"
MINI_SOURCE_PC_RANGE="${MINI_SOURCE_PC_RANGE:-${DEFAULT_MINI_SOURCE_PC_RANGE}}"
MINI_MODEL_PC_RANGE="${MINI_MODEL_PC_RANGE:-${DEFAULT_MINI_MODEL_PC_RANGE}}"
MINI_RADAR_NORMALIZATION_PATH="${MINI_RADAR_NORMALIZATION_PATH:-${PROJECT_DIR}/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json}"
MINI_DOPPLER_SCALE_MPS="${MINI_DOPPLER_SCALE_MPS:-86.8}"
MINI_CHECKPOINT_PROTOCOL="${MINI_CHECKPOINT_PROTOCOL:-formal_mini_chain_v2}"
EXPECTED_FORMAL_ARTIFACT_SHA256="${EXPECTED_FORMAL_ARTIFACT_SHA256:-11f59d84cc186c39256c112154faf458ec9ead5fec9b08b997abd5058b68e97c}"
MINI_TEMPORAL_SPLIT_ARTIFACT="${MINI_TEMPORAL_SPLIT_ARTIFACT:-${PREPROCESSED_ROOT}/temporal_split_garden_train80_purge3s_v1.json}"
MINI_DATA_PROTOCOL_PATH="${MINI_DATA_PROTOCOL_PATH:-${PREPROCESSED_ROOT}/formal_data_protocol_garden_train80_purge3s_v1.json}"
MINI_TRAIN_FRAMES_PER_SCENE="${MINI_TRAIN_FRAMES_PER_SCENE:-8}"
MINI_VALIDATION_FRAMES_PER_SCENE="${MINI_VALIDATION_FRAMES_PER_SCENE:-4}"
MINI_VAE_CONFIG_TYPE="${MINI_VAE_CONFIG_TYPE:-ultra_lightweight}"
MINI_VAE_LATENT_DIM="${MINI_VAE_LATENT_DIM:-}"
MINI_VAE_OCC_LOSS="${MINI_VAE_OCC_LOSS:-bce_dice}"
MINI_TRAIN_SPLIT="${MINI_TRAIN_SPLIT:-0.8}"
MINI_SPLIT_SEED="${MINI_SPLIT_SEED:-42}"
MINI_LDM_DECODED_WEIGHT="${MINI_LDM_DECODED_WEIGHT:-}"
MINI_LDM_DECODED_FP_WEIGHT="${MINI_LDM_DECODED_FP_WEIGHT:-}"
MINI_LDM_DECODED_MASS_WEIGHT="${MINI_LDM_DECODED_MASS_WEIGHT:-}"
MINI_LDM_HEIGHT_WEIGHT="${MINI_LDM_HEIGHT_WEIGHT:-0.02}"
MINI_LDM_TOP_WEIGHT="${MINI_LDM_TOP_WEIGHT:-0.0}"
MINI_LDM_TOP_OVERSHOOT_WEIGHT="${MINI_LDM_TOP_OVERSHOOT_WEIGHT:-0.0}"
MINI_LDM_CONTINUITY_WEIGHT="${MINI_LDM_CONTINUITY_WEIGHT:-0.02}"
MINI_LDM_DENSITY_WEIGHT="${MINI_LDM_DENSITY_WEIGHT:-0.0}"
MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT:-0.0}"
MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT:-0.0}"
MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT:-0.0}"
MINI_LDM_UNCERTAINTY_WEIGHT="${MINI_LDM_UNCERTAINTY_WEIGHT:-}"
MINI_LDM_COLUMN_CURRICULUM_ENABLED="${MINI_LDM_COLUMN_CURRICULUM_ENABLED:-false}"
MINI_LDM_COLUMN_POSITIVE_START_WEIGHT="${MINI_LDM_COLUMN_POSITIVE_START_WEIGHT:-0.0}"
MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT="${MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT:-0.0}"
MINI_LDM_COLUMN_POSITIVE_WEIGHT="${MINI_LDM_COLUMN_POSITIVE_WEIGHT:-0.0}"
MINI_LDM_COLUMN_NEGATIVE_WEIGHT="${MINI_LDM_COLUMN_NEGATIVE_WEIGHT:-0.0}"
MINI_LDM_COLUMN_TEMPERATURE="${MINI_LDM_COLUMN_TEMPERATURE:-1.0}"

MODE="${1:-all}"
if [[ -n "${CUDA_DEVICES:-}" ]]; then
	SELECTED_CUDA_DEVICES="${CUDA_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	SELECTED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
	SELECTED_CUDA_DEVICES="0"
fi
export CUDA_DEVICES="${SELECTED_CUDA_DEVICES}"
export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_DEVICES}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-200}"

MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-2}"
MINI_GRAD_ACCUM="${MINI_GRAD_ACCUM:-1}"
MINI_USE_AUG="${MINI_USE_AUG:-false}"

MINI_VAE_EPOCHS="${MINI_VAE_EPOCHS:-3}"
MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-2}"
MINI_CD_EPOCHS="${MINI_CD_EPOCHS:-1}"

validate_mode() {
	case "${MODE}" in
		vae|ldm|cd|all|all_with_cd)
			;;
		*)
			echo "Usage: $0 [vae|ldm|cd|all|all_with_cd]"
			exit 1
			;;
	esac
}

validate_mini_dataset_dir() {
	local normalized_path
	local normalized_preprocessed_root
	local normalized_results_dir

	normalized_path="$(realpath -m -- "${MINI_DATASET_DIR}")"
	normalized_preprocessed_root="$(realpath -m -- "${PREPROCESSED_ROOT}")"
	normalized_results_dir="$(realpath -m -- "${MINI_RESULTS_DIR}")"
	MINI_RESULTS_DIR="${normalized_results_dir}"
	MINI_DATASET_DIR="${normalized_path}"

	if [[ "${MINI_RADAR_PROTOCOL}" == "formal" ]]; then
		if [[ "${MINI_REQUIRE_FRESH_SCRATCH}" != "0" ]]; then
			echo "Error: formal v2 mini 直接只读正式数据根，不允许 fresh scratch"
			exit 1
		fi
		if [[ "${normalized_path}" != "${normalized_preprocessed_root}" ]]; then
			echo "Error: formal v2 MINI_DATASET_DIR 必须等于 PREPROCESSED_ROOT"
			exit 1
		fi
		if [[ -L "${MINI_DATASET_DIR_INPUT}" || ! -d "${MINI_DATASET_DIR_INPUT}" ]]; then
			echo "Error: formal v2 数据根必须是现有普通目录: ${MINI_DATASET_DIR_INPUT}"
			exit 1
		fi
		return 0
	fi

	if [[ "${MINI_REQUIRE_FRESH_SCRATCH}" == "1" ]]; then
		if [[ -e "${MINI_DATASET_DIR_INPUT}" || -L "${MINI_DATASET_DIR_INPUT}" ]]; then
			echo "Error: fresh MINI_DATASET_DIR already exists or is a symlink: ${MINI_DATASET_DIR_INPUT}"
			exit 1
		fi
		case "${normalized_path}" in
			"${normalized_results_dir}"/*) ;;
			*)
				echo "Error: fresh MINI_DATASET_DIR must be inside MINI_RESULTS_DIR: ${normalized_path}"
				exit 1
				;;
		esac
	fi
	if [[ "${MINI_REQUIRE_FRESH_CONFIG}" == "1" ]]; then
		local normalized_config_path
		local normalized_config_parent
		if [[ -e "${MINI_CONFIG_PATH_INPUT}" || -L "${MINI_CONFIG_PATH_INPUT}" ]]; then
			echo "Error: fresh MINI_CONFIG_PATH already exists or is a symlink: ${MINI_CONFIG_PATH_INPUT}"
			exit 1
		fi
		normalized_config_path="$(realpath -m -- "${MINI_CONFIG_PATH_INPUT}")"
		normalized_config_parent="$(realpath -m -- "$(dirname -- "${normalized_config_path}")")"
		case "${normalized_config_parent}" in
			"${normalized_results_dir}" | "${normalized_results_dir}"/*) ;;
			*)
				echo "Error: fresh MINI_CONFIG_PATH parent must be inside MINI_RESULTS_DIR: ${normalized_config_parent}"
				exit 1
				;;
		esac
		MINI_CONFIG_PATH="${normalized_config_path}"
	fi
	if [[ "${MINI_DATASET_DIR}" == "/" ||
		"${MINI_DATASET_DIR}" == "/tmp" ||
		"${MINI_DATASET_DIR}" == "${ROOT_DIR}" ||
		"${MINI_DATASET_DIR}" == "${normalized_preprocessed_root}" ||
		"${MINI_DATASET_DIR}" == "${normalized_results_dir}" ]]; then
		echo "Error: unsafe MINI_DATASET_DIR: ${MINI_DATASET_DIR}"
		exit 1
	fi

	if [[ "${MINI_DATASET_DIR}" == /tmp/* ]]; then
		return 0
	fi
	if [[ "${MINI_DATASET_DIR}" == "${ROOT_DIR}/test/"* ]] &&
		[[ "$(basename "${MINI_DATASET_DIR}")" == .tmp_* ]]; then
		return 0
	fi

	echo "Error: unsafe MINI_DATASET_DIR: ${MINI_DATASET_DIR}"
	echo "Allowed paths: a non-root path under /tmp, or ROOT_DIR/test/**/.tmp_*"
	exit 1
}

validate_mini_radar_protocol() {
	if [[ "${MINI_RADAR_PROTOCOL}" == "legacy" ]]; then
		return 0
	fi
	if [[ "${MINI_CHECKPOINT_PROTOCOL}" != "formal_mini_chain_v2" ]]; then
		echo "Error: formal mini checkpoint protocol must be formal_mini_chain_v2"
		exit 1
	fi
	if [[ ! -f "${MINI_RADAR_NORMALIZATION_PATH}" ]]; then
		echo "Error: formal mini normalization artifact not found: ${MINI_RADAR_NORMALIZATION_PATH}"
		exit 1
	fi

	"${PYTHON_CMD[@]}" - \
		"${ROOT_DIR}" \
		"${MINI_RADAR_NORMALIZATION_PATH}" \
		"${EXPECTED_FORMAL_ARTIFACT_SHA256}" \
		"${MINI_TARGET_SIZE}" \
		"${MINI_SOURCE_PC_RANGE}" \
		"${MINI_MODEL_PC_RANGE}" \
		"${MINI_DOPPLER_SCALE_MPS}" \
		"${PREPROCESSED_ROOT}" \
		"${MINI_TEMPORAL_SPLIT_ARTIFACT}" \
		"${MINI_DATA_PROTOCOL_PATH}" \
		"${MODE}" \
		"${MINI_RESULTS_DIR}" \
		"${MINI_TRAIN_FRAMES_PER_SCENE}" \
		"${MINI_VALIDATION_FRAMES_PER_SCENE}" \
		"${TRAIN_SCENES[@]}" <<'PY'
import hashlib
import os
import sys

(
    root,
    artifact,
    expected_sha256,
    target_raw,
    source_raw,
    model_raw,
    scale_raw,
    dataset_dir,
    split_path,
    data_protocol_path,
    stage,
    results_dir,
    train_limit_raw,
    validation_limit_raw,
    *scenes,
) = sys.argv[1:]
if root not in sys.path:
    sys.path.insert(0, root)

from diffusion_consistency_radar.formal_data_protocol import (
    load_formal_data_protocol_artifact,
)
from diffusion_consistency_radar.checkpoint_chain import (
    assert_checkpoint_training_identity,
    build_formal_mini_selection,
    checkpoint_state_dict,
    safe_torch_load,
    sha256_file,
)
from diffusion_consistency_radar.radar_normalization import (
    load_radar_normalization_artifact,
)
from diffusion_consistency_radar.temporal_split import (
    limit_frame_ids_by_scene,
    load_temporal_split_artifact,
    split_frame_ids_by_scene,
)

target_size = [int(value) for value in target_raw.split(',')]
source_pc_range = [float(value) for value in source_raw.split(',')]
model_pc_range = [float(value) for value in model_raw.split(',')]
with open(artifact, "rb") as handle:
    artifact_file_sha256 = hashlib.sha256(handle.read()).hexdigest()
if artifact_file_sha256 != expected_sha256:
    raise SystemExit(
        "formal mini normalization SHA-256 mismatch: "
        f"expected={expected_sha256}, actual={artifact_file_sha256}"
    )
split_artifact, split_sha256 = load_temporal_split_artifact(
    split_path,
    dataset_dir=dataset_dir,
    expected_scenes=scenes,
    require_formal=True,
)
data_protocol, _data_protocol_sha256 = load_formal_data_protocol_artifact(
    data_protocol_path,
    dataset_dir=dataset_dir,
    scenes=scenes,
    split_artifact_path=split_path,
    stage=stage,
)
if data_protocol.get("split_artifact_sha256") != split_sha256:
    raise SystemExit("formal mini data protocol 与 temporal split SHA-256 不一致")
train_ids = limit_frame_ids_by_scene(
    split_frame_ids_by_scene(split_artifact, "train"),
    int(train_limit_raw),
    partition="train",
)
validation_ids = limit_frame_ids_by_scene(
    split_frame_ids_by_scene(split_artifact, "validation"),
    int(validation_limit_raw),
    partition="validation",
)
current_data_protocol = dict(data_protocol)
current_data_protocol["mini_selection"] = build_formal_mini_selection(
    int(train_limit_raw),
    int(validation_limit_raw),
)

# LDM/CD 的无训练预检必须和真正训练入口使用同一身份断言，避免把错误父权重
# 延迟到 CUDA 训练进程启动后才发现。VAE stage 没有父 checkpoint。
parent_specs = []
if stage in {"ldm", "cd"}:
    parent_specs.append(("vae", os.path.join(results_dir, "vae", "vae_best.pt")))
if stage == "cd":
    parent_specs.append(("ldm", os.path.join(results_dir, "ldm", "ldm_best.pt")))
parent_hashes = {}
parent_checkpoints = {}
for parent_stage, parent_path in parent_specs:
    if os.path.islink(parent_path) or not os.path.isfile(parent_path):
        raise SystemExit(
            f"formal mini {parent_stage} parent 必须是现有普通文件: {parent_path}"
        )
    checkpoint = safe_torch_load(parent_path, map_location="cpu")
    assert_checkpoint_training_identity(
        checkpoint,
        expected_stage=parent_stage,
        checkpoint_protocol="formal_mini_chain_v2",
        data_protocol=current_data_protocol,
    )
    checkpoint_state_dict(checkpoint)
    parent_checkpoints[parent_stage] = checkpoint
    parent_hashes[parent_stage] = sha256_file(parent_path)
if stage == "cd":
    if parent_checkpoints["ldm"].get("vae_checkpoint_sha256") != parent_hashes["vae"]:
        raise SystemExit(
            "formal mini LDM 记录的 vae_checkpoint_sha256 与当前 VAE 文件不一致"
        )
if parent_specs:
    print(
        "Formal mini parent checkpoint validated: "
        + ", ".join(
            f"{parent_stage}={parent_hashes[parent_stage]}"
            for parent_stage, _parent_path in parent_specs
        )
    )
_spec, actual_sha256 = load_radar_normalization_artifact(
    artifact,
    target_size=target_size,
    source_pc_range=source_pc_range,
    model_pc_range=model_pc_range,
    doppler_scale_mps=float(scale_raw),
    require_formal=True,
    expected_split_artifact_sha256=split_sha256,
)
if actual_sha256 != expected_sha256:
    raise SystemExit(
        "formal mini normalization SHA-256 mismatch: "
        f"expected={expected_sha256}, actual={actual_sha256}"
    )
print(f"Formal mini Radar normalization validated: {actual_sha256}")
print(
    "Formal mini v2 data protocol validated: "
    f"train={sum(map(len, train_ids.values()))}, "
    f"validation={sum(map(len, validation_ids.values()))}"
)
PY
}

validate_mode
validate_mini_dataset_dir
if [[ "${MINI_PREFLIGHT_ONLY}" != "0" && "${MINI_PREFLIGHT_ONLY}" != "1" ]]; then
	echo "Error: MINI_PREFLIGHT_ONLY must be 0 or 1"
	exit 1
fi
if [[ "${MINI_RADAR_PROTOCOL}" == "formal" ]]; then
	for setting in MINI_TRAIN_FRAMES_PER_SCENE MINI_VALIDATION_FRAMES_PER_SCENE; do
		value="${!setting}"
		if [[ ! "${value}" =~ ^[0-9]+$ || "${value}" -lt 1 ]]; then
			echo "Error: ${setting} must be a positive integer"
			exit 1
		fi
	done
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON_CMD=("${PYTHON_BIN}")
elif python -c "import torch" >/dev/null 2>&1; then
	PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
	# conda run 默认捕获模式不会可靠转发 heredoc stdin；正式 artifact 校验依赖 stdin 脚本。
	PYTHON_CMD=(conda run --no-capture-output -n Radar-Diffusion python)
else
	PYTHON_CMD=(python)
fi

if python3 -c "import yaml" >/dev/null 2>&1; then
	CONFIG_PYTHON_CMD=(python3)
elif python -c "import yaml" >/dev/null 2>&1; then
	CONFIG_PYTHON_CMD=(python)
else
	CONFIG_PYTHON_CMD=("${PYTHON_CMD[@]}")
fi

if [[ ! -f "${DATA_LOADING_CONFIG}" ]]; then
	echo "Error: data loading config not found: ${DATA_LOADING_CONFIG}"
	exit 1
fi

if [[ -n "${TRAIN_SCENES_OVERRIDE:-}" ]]; then
	IFS=',' read -r -a TRAIN_SCENES <<< "${TRAIN_SCENES_OVERRIDE}"
elif [[ -n "${SCENE:-}" ]]; then
	TRAIN_SCENES=("${SCENE}")
else
mapfile -t TRAIN_SCENES < <("${CONFIG_PYTHON_CMD[@]}" - "${DATA_LOADING_CONFIG}" <<'PY'
import os
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
fi

if [[ ${#TRAIN_SCENES[@]} -eq 0 ]]; then
	echo "Error: data.train is empty in ${DATA_LOADING_CONFIG}"
	exit 1
fi

for SCENE in "${TRAIN_SCENES[@]}"; do
	SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
	if [[ ! -d "${SRC_SCENE_DIR}/radar_voxel" || ! -d "${SRC_SCENE_DIR}/target_voxel" ]]; then
		echo "Error: missing radar_voxel/target_voxel in ${SRC_SCENE_DIR}"
		exit 1
	fi
	if [[ "${MINI_RADAR_PROTOCOL}" == "formal" && ! -d "${SRC_SCENE_DIR}/ir_image" ]]; then
		echo "Error: formal mini missing ir_image in ${SRC_SCENE_DIR}"
		exit 1
	fi
done

validate_mini_radar_protocol

echo "=========================================="
echo "Minimal training setup"
echo "mode: ${MODE}"
echo "train scenes: ${TRAIN_SCENES[*]}"
if [[ "${MINI_RADAR_PROTOCOL}" == "legacy" ]]; then
	echo "legacy samples per scene: ${SAMPLES_PER_SCENE}"
fi
echo "mini epochs: vae=${MINI_VAE_EPOCHS}, ldm=${MINI_LDM_EPOCHS}, cd=${MINI_CD_EPOCHS}"
echo "project dir: ${PROJECT_DIR}"
echo "preprocessed root: ${PREPROCESSED_ROOT}"
echo "results dir: ${MINI_RESULTS_DIR}"
echo "training dataset dir: ${MINI_DATASET_DIR}"
echo "target size [Z,X,Y]: ${MINI_TARGET_SIZE}"
echo "source pc range: ${MINI_SOURCE_PC_RANGE}"
echo "model pc range: ${MINI_MODEL_PC_RANGE}"
echo "radar protocol: ${MINI_RADAR_PROTOCOL}"
echo "radar normalization: ${MINI_RADAR_NORMALIZATION_PATH}"
echo "checkpoint protocol: ${MINI_CHECKPOINT_PROTOCOL}"
if [[ "${MINI_RADAR_PROTOCOL}" == "formal" ]]; then
	echo "temporal split: ${MINI_TEMPORAL_SPLIT_ARTIFACT}"
	echo "formal data protocol: ${MINI_DATA_PROTOCOL_PATH}"
	echo "formal mini train/validation frames per scene: ${MINI_TRAIN_FRAMES_PER_SCENE}/${MINI_VALIDATION_FRAMES_PER_SCENE}"
fi
echo "vae config type: ${MINI_VAE_CONFIG_TYPE}"
echo "vae latent dim: ${MINI_VAE_LATENT_DIM:-preset}"
echo "vae occupancy loss: ${MINI_VAE_OCC_LOSS}"
echo "train split: ${MINI_TRAIN_SPLIT}"
echo "split seed: ${MINI_SPLIT_SEED}"
echo "ldm decoded occupancy weight: ${MINI_LDM_DECODED_WEIGHT:-config default}"
echo "ldm decoded false-positive weight: ${MINI_LDM_DECODED_FP_WEIGHT:-config default}"
echo "ldm decoded mass weight: ${MINI_LDM_DECODED_MASS_WEIGHT:-config default}"
echo "ldm height distribution weight: ${MINI_LDM_HEIGHT_WEIGHT}"
echo "ldm top height weight: ${MINI_LDM_TOP_WEIGHT}"
echo "ldm top overshoot weight: ${MINI_LDM_TOP_OVERSHOOT_WEIGHT}"
echo "ldm vertical continuity weight: ${MINI_LDM_CONTINUITY_WEIGHT}"
echo "ldm density weight: ${MINI_LDM_DENSITY_WEIGHT}"
echo "ldm IR frustum occupancy/top weights: ${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT}/${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT}"
echo "ldm IR frustum negative weight: ${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT}"
echo "ldm uncertainty weight: ${MINI_LDM_UNCERTAINTY_WEIGHT:-config default}"
echo "ldm column curriculum enabled: ${MINI_LDM_COLUMN_CURRICULUM_ENABLED}"
echo "ldm column positive/negative start weights: ${MINI_LDM_COLUMN_POSITIVE_START_WEIGHT}/${MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT}"
echo "ldm column positive/negative weights: ${MINI_LDM_COLUMN_POSITIVE_WEIGHT}/${MINI_LDM_COLUMN_NEGATIVE_WEIGHT}"
echo "ldm column temperature: ${MINI_LDM_COLUMN_TEMPERATURE}"
echo "=========================================="

if [[ "${MINI_PREFLIGHT_ONLY}" == "1" ]]; then
	echo "Mini training preflight passed; no scratch/config/output was created."
	exit 0
fi

if [[ "${MINI_RADAR_PROTOCOL}" == "legacy" ]]; then
if [[ "${MINI_REQUIRE_FRESH_SCRATCH}" == "1" ]]; then
	mkdir -- "${MINI_DATASET_DIR}"
	CREATED_DATASET_DIR="$(realpath -m -- "${MINI_DATASET_DIR}")"
	case "${CREATED_DATASET_DIR}" in
		"${MINI_RESULTS_DIR}"/*) ;;
		*) echo "Error: created MINI_DATASET_DIR escaped MINI_RESULTS_DIR: ${CREATED_DATASET_DIR}"; exit 1 ;;
	esac
	if [[ "${CREATED_DATASET_DIR}" != "${MINI_DATASET_DIR}" || -L "${MINI_DATASET_DIR}" ]]; then
		echo "Error: created MINI_DATASET_DIR changed unexpectedly: ${CREATED_DATASET_DIR}"
		exit 1
	fi
else
	rm -rf "${MINI_DATASET_DIR}"
	mkdir -p "${MINI_DATASET_DIR}"
fi
if [[ -d "${CALIB_CONFIG_DIR}" ]]; then
	ln -s "${CALIB_CONFIG_DIR}" "${MINI_DATASET_DIR}/config"
else
	echo "Warning: calibration config directory not found: ${CALIB_CONFIG_DIR}"
fi

for SCENE in "${TRAIN_SCENES[@]}"; do
	SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
	SRC_RADAR_DIR="${SRC_SCENE_DIR}/radar_voxel"
	SRC_TARGET_DIR="${SRC_SCENE_DIR}/target_voxel"
	SRC_IR_DIR="${SRC_SCENE_DIR}/ir_image"
	DST_SCENE_DIR="${MINI_DATASET_DIR}/${SCENE}"
	DST_RADAR_DIR="${DST_SCENE_DIR}/radar_voxel"
	DST_TARGET_DIR="${DST_SCENE_DIR}/target_voxel"
	DST_IR_DIR="${DST_SCENE_DIR}/ir_image"

	if [[ ! -d "${SRC_RADAR_DIR}" || ! -d "${SRC_TARGET_DIR}" ]]; then
		echo "Error: missing radar_voxel/target_voxel in ${SRC_SCENE_DIR}"
		exit 1
	fi

	mkdir -p "${DST_RADAR_DIR}" "${DST_TARGET_DIR}" "${DST_IR_DIR}"
	if [[ -f "${SRC_SCENE_DIR}/preprocess_policy.json" ]]; then
		ln -s "${SRC_SCENE_DIR}/preprocess_policy.json" "${DST_SCENE_DIR}/preprocess_policy.json"
	fi

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

		FRAME_STEM="${FILE_NAME%.*}"
		SRC_IR_PATH="${SRC_IR_DIR}/${FRAME_STEM}_ir.npy"
		if [[ -f "${SRC_IR_PATH}" ]]; then
			ln -s "${SRC_IR_PATH}" "${DST_IR_DIR}/${FRAME_STEM}_ir.npy"
		fi
	done
done
fi

mkdir -p "${MINI_RESULTS_DIR}/vae" "${MINI_RESULTS_DIR}/ldm" "${MINI_RESULTS_DIR}/cd"

OLD_IFS="${IFS}"
IFS=,
TRAIN_SCENES_CSV="${TRAIN_SCENES[*]}"
IFS="${OLD_IFS}"

"${CONFIG_PYTHON_CMD[@]}" - "${DEFAULT_CONFIG_PATH}" "${MINI_CONFIG_PATH}" "${MINI_DATASET_DIR}" "${MINI_BATCH_SIZE}" "${MINI_NUM_WORKERS}" "${MINI_USE_AUG}" "${MINI_VAE_EPOCHS}" "${MINI_LDM_EPOCHS}" "${MINI_CD_EPOCHS}" "${MINI_GRAD_ACCUM}" "${MINI_RESULTS_DIR}" "${MINI_TARGET_SIZE}" "${MINI_SOURCE_PC_RANGE}" "${MINI_MODEL_PC_RANGE}" "${MINI_RADAR_PROTOCOL}" "${MINI_RADAR_NORMALIZATION_PATH}" "${MINI_DOPPLER_SCALE_MPS}" "${MINI_CHECKPOINT_PROTOCOL}" "${MINI_TEMPORAL_SPLIT_ARTIFACT}" "${MINI_DATA_PROTOCOL_PATH}" "${MINI_TRAIN_FRAMES_PER_SCENE}" "${MINI_VALIDATION_FRAMES_PER_SCENE}" "${CALIB_CONFIG_DIR}" "${TRAIN_SCENES_CSV}" "${MINI_VAE_CONFIG_TYPE}" "${MINI_VAE_LATENT_DIM}" "${MINI_VAE_OCC_LOSS}" "${MINI_TRAIN_SPLIT}" "${MINI_SPLIT_SEED}" "${MINI_LDM_DECODED_WEIGHT}" "${MINI_LDM_DECODED_FP_WEIGHT}" "${MINI_LDM_DECODED_MASS_WEIGHT}" "${MINI_LDM_HEIGHT_WEIGHT}" "${MINI_LDM_TOP_WEIGHT}" "${MINI_LDM_TOP_OVERSHOOT_WEIGHT}" "${MINI_LDM_CONTINUITY_WEIGHT}" "${MINI_LDM_DENSITY_WEIGHT}" "${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT}" "${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT}" "${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT}" "${MINI_LDM_UNCERTAINTY_WEIGHT}" "${MINI_LDM_COLUMN_CURRICULUM_ENABLED}" "${MINI_LDM_COLUMN_POSITIVE_START_WEIGHT}" "${MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT}" "${MINI_LDM_COLUMN_POSITIVE_WEIGHT}" "${MINI_LDM_COLUMN_NEGATIVE_WEIGHT}" "${MINI_LDM_COLUMN_TEMPERATURE}" "${MINI_REQUIRE_FRESH_CONFIG}" <<'PY'
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
		cd_epochs,
		grad_accum,
		results_dir,
		target_size_raw,
		source_pc_range_raw,
		model_pc_range_raw,
		radar_protocol,
		radar_normalization_path,
		doppler_scale_mps,
		checkpoint_protocol,
		temporal_split_artifact,
		data_protocol_path,
		mini_train_frames_per_scene,
		mini_validation_frames_per_scene,
		calibration_dir,
		scene_names_csv,
		vae_config_type,
		vae_latent_dim,
		vae_occ_loss,
		train_split,
		split_seed,
		ldm_decoded_weight,
		ldm_decoded_fp_weight,
		ldm_decoded_mass_weight,
		ldm_height_weight,
		ldm_top_weight,
		ldm_top_overshoot_weight,
		ldm_continuity_weight,
		ldm_density_weight,
		ldm_ir_frustum_occ_weight,
		ldm_ir_frustum_negative_weight,
		ldm_ir_frustum_top_weight,
		ldm_uncertainty_weight,
		ldm_column_curriculum_enabled,
		ldm_column_positive_start_weight,
		ldm_column_negative_start_weight,
		ldm_column_positive_weight,
		ldm_column_negative_weight,
		ldm_column_temperature,
		require_fresh_config,
) = sys.argv[1:49]


def parse_numbers(raw, caster):
		return [caster(v.strip()) for v in str(raw).split(',') if v.strip()]


def parse_strict_bool(raw, variable_name):
		value = str(raw).strip().lower()
		if value in {'1', 'true', 'yes', 'on'}:
				return True
		if value in {'0', 'false', 'no', 'off'}:
				return False
		raise SystemExit(
				f"{variable_name} must be one of 1,true,yes,on,0,false,no,off; got {raw!r}"
		)

with open(src_cfg, 'r', encoding='utf-8') as f:
			cfg = yaml.safe_load(f) or {}

allocator_conf = __import__('os').environ.get('PYTORCH_CUDA_ALLOC_CONF', '').strip()
if allocator_conf:
			cfg.setdefault('hardware', {})
			cfg['hardware']['cuda_allocator_conf'] = allocator_conf

target_size = parse_numbers(target_size_raw, int)
source_pc_range = parse_numbers(source_pc_range_raw, float)
model_pc_range = parse_numbers(model_pc_range_raw, float)
if len(target_size) != 3:
		raise SystemExit(f"MINI_TARGET_SIZE must contain 3 values, got {target_size_raw}")
if len(source_pc_range) != 6 or len(model_pc_range) != 6:
		raise SystemExit("MINI_SOURCE_PC_RANGE and MINI_MODEL_PC_RANGE must contain 6 values")

cfg.setdefault('data', {})
cfg['data']['dataset_dir'] = dataset_dir
cfg['data']['batch_size'] = int(batch_size)
cfg['data']['num_workers'] = int(num_workers)
cfg['data']['use_augmentation'] = str(use_aug).lower() in {'1', 'true', 'yes', 'on'}
cfg['data']['target_size'] = target_size
cfg['data']['source_pc_range'] = source_pc_range
cfg['data']['model_pc_range'] = model_pc_range
cfg['data']['train_split'] = float(train_split)
cfg['data']['split_seed'] = int(split_seed)
if radar_protocol == 'formal':
		if checkpoint_protocol != 'formal_mini_chain_v2':
			raise SystemExit('formal mini checkpoint protocol must be formal_mini_chain_v2')
		scene_names = [value.strip() for value in scene_names_csv.split(',') if value.strip()]
		if not scene_names:
			raise SystemExit('formal mini scene_names must not be empty')
		cfg['data']['radar_normalization_path'] = radar_normalization_path
		cfg['data']['doppler_scale_mps'] = float(doppler_scale_mps)
		cfg['data']['checkpoint_protocol'] = checkpoint_protocol
		cfg['data']['temporal_split_artifact'] = temporal_split_artifact
		cfg['data']['data_protocol_path'] = data_protocol_path
		cfg['data'].pop('data_protocol', None)
		cfg['data']['mini_train_frames_per_scene'] = int(mini_train_frames_per_scene)
		cfg['data']['mini_validation_frames_per_scene'] = int(mini_validation_frames_per_scene)
		cfg['data']['scene_names'] = scene_names
		cfg['data']['calibration_dir'] = calibration_dir
		cfg['data']['require_real_ir'] = True
		cfg['data']['require_real_calibration'] = True
		cfg['data']['require_persisted_observed_mask'] = True
		cfg['data']['require_radar_statistics'] = True
		cfg['data']['voxel_coordinate_frame'] = 'lidar'
		cfg.setdefault('hardware', {})
		cfg['hardware']['device'] = 'cuda'
		cfg['hardware']['num_gpus'] = 1
elif radar_protocol == 'legacy':
		# legacy mini 必须移除从正式默认配置继承的 artifact。
		cfg['data']['radar_normalization_path'] = ''
		cfg['data']['doppler_scale_mps'] = None
		cfg['data'].pop('checkpoint_protocol', None)
		for key in (
				'temporal_split_artifact', 'data_protocol_path', 'data_protocol',
				'mini_train_frames_per_scene', 'mini_validation_frames_per_scene',
				'scene_names', 'calibration_dir', 'require_real_ir',
				'require_real_calibration', 'require_persisted_observed_mask',
				'require_radar_statistics', 'voxel_coordinate_frame',
		):
				cfg['data'].pop(key, None)
else:
		raise SystemExit(f'unsupported MINI_RADAR_PROTOCOL={radar_protocol!r}')

cfg['vae'] = dict(cfg.get('vae') or {})
cfg['vae']['epochs'] = int(vae_epochs)
cfg['vae']['save_every'] = 1
cfg['vae']['save_dir'] = f"{results_dir}/vae"
cfg['vae']['config_type'] = vae_config_type
if vae_latent_dim:
		cfg['vae']['latent_dim'] = int(vae_latent_dim)
else:
		cfg['vae'].pop('latent_dim', None)
cfg['vae']['occupancy_loss_type'] = vae_occ_loss

cfg['ldm'] = dict(cfg.get('ldm') or {})
cfg['ldm']['epochs'] = int(ldm_epochs)
cfg['ldm']['save_every'] = 1
cfg['ldm']['save_dir'] = f"{results_dir}/ldm"
if ldm_decoded_weight:
		cfg['ldm']['decoded_loss_weight'] = float(ldm_decoded_weight)
if ldm_decoded_fp_weight:
		cfg['ldm']['decoded_false_positive_weight'] = float(ldm_decoded_fp_weight)
if ldm_decoded_mass_weight:
		cfg['ldm']['decoded_mass_weight'] = float(ldm_decoded_mass_weight)
if ldm_uncertainty_weight:
		cfg['ldm']['uncertainty_loss_weight'] = float(ldm_uncertainty_weight)
else:
		cfg['ldm']['uncertainty_loss_weight'] = float(cfg['ldm'].get('uncertainty_loss_weight', 0.05))
cfg['ldm']['decoded_height_distribution_weight'] = float(ldm_height_weight)
cfg['ldm']['decoded_top_height_weight'] = float(ldm_top_weight)
cfg['ldm']['decoded_top_overshoot_weight'] = float(ldm_top_overshoot_weight)
cfg['ldm']['decoded_vertical_continuity_weight'] = float(ldm_continuity_weight)
cfg['ldm']['decoded_density_weight'] = float(ldm_density_weight)
cfg['ldm']['decoded_ir_frustum_occupancy_weight'] = float(ldm_ir_frustum_occ_weight)
cfg['ldm']['decoded_ir_frustum_negative_weight'] = float(ldm_ir_frustum_negative_weight)
cfg['ldm']['decoded_ir_frustum_top_weight'] = float(ldm_ir_frustum_top_weight)
cfg['ldm']['decoded_column_curriculum_enabled'] = parse_strict_bool(ldm_column_curriculum_enabled, 'MINI_LDM_COLUMN_CURRICULUM_ENABLED')
cfg['ldm']['decoded_column_positive_start_weight'] = float(ldm_column_positive_start_weight)
cfg['ldm']['decoded_column_negative_start_weight'] = float(ldm_column_negative_start_weight)
cfg['ldm']['decoded_column_positive_weight'] = float(ldm_column_positive_weight)
cfg['ldm']['decoded_column_negative_weight'] = float(ldm_column_negative_weight)
cfg['ldm']['decoded_column_temperature'] = float(ldm_column_temperature)
cfg['ldm']['fusion_voxel_shape'] = target_size
cfg['ldm']['fusion_pc_range'] = model_pc_range

cfg['cd'] = {
	'initialization_model_path': f"{results_dir}/ldm/ldm_best.pt",
	'teacher_model_path': '',
	'training_semantics': 'ldm_initialized_ema_consistency_v1',
	'num_scales': 40,
	'ema_rate': 0.999,
	'sigma_min': 0.002,
	'sigma_max': 80.0,
	'rho': 7.0,
	'epochs': int(cd_epochs),
	'lr': 5.0e-5,
	'save_every': 1,
	'save_dir': f"{results_dir}/cd",
}

cfg.setdefault('optimization', {})
cfg['optimization']['gradient_accumulation_steps'] = int(grad_accum)
if radar_protocol == 'formal':
	# 8 GB formal mini 显式保留 checkpointing；AMP 当前因 VAE dtype 接口不兼容而关闭。
	cfg['optimization']['use_checkpoint'] = True
	cfg['optimization']['use_amp'] = False
	cfg['optimization']['use_fp16'] = False

config_write_mode = 'x' if str(require_fresh_config).lower() in {'1', 'true', 'yes', 'on'} else 'w'
with open(dst_cfg, config_write_mode, encoding='utf-8') as f:
		yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

echo "Mini config: ${MINI_CONFIG_PATH}"
echo "Mini dataset: ${MINI_DATASET_DIR}"

RADAR_PROTOCOL_ARGS=()
if [[ "${MINI_RADAR_PROTOCOL}" == "legacy" ]]; then
	RADAR_PROTOCOL_ARGS+=(--allow_legacy_radar_units)
fi

run_vae() {
	CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/unified_train.py" \
		--mode vae \
		--config "${MINI_CONFIG_PATH}" \
		"${RADAR_PROTOCOL_ARGS[@]}"
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
		--vae_ckpt "${vae_ckpt}" \
		"${RADAR_PROTOCOL_ARGS[@]}"
}

run_cd() {
	local vae_ckpt="${MINI_RESULTS_DIR}/vae/vae_best.pt"
	local ldm_ckpt="${MINI_RESULTS_DIR}/ldm/ldm_best.pt"
	if [[ ! -f "${vae_ckpt}" ]]; then
		echo "Error: minimal VAE checkpoint not found: ${vae_ckpt}"
		echo "Run VAE first: bash ${SELF_DIR}/train_minimal.sh vae"
		exit 1
	fi
	if [[ ! -f "${ldm_ckpt}" ]]; then
		echo "Error: minimal LDM checkpoint not found: ${ldm_ckpt}"
		echo "Run LDM first: bash ${SELF_DIR}/train_minimal.sh ldm"
		exit 1
	fi

	CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/unified_train.py" \
		--mode cd \
		--config "${MINI_CONFIG_PATH}" \
		--vae_ckpt "${vae_ckpt}" \
		--ldm_ckpt "${ldm_ckpt}" \
		"${RADAR_PROTOCOL_ARGS[@]}"
}

case "${MODE}" in
	vae)
		run_vae
		;;
	ldm)
		run_ldm
		;;
	cd)
		run_cd
		;;
	all)
		run_vae
		run_ldm
		;;
	all_with_cd)
		run_vae
		run_ldm
		run_cd
		;;
	*)
		echo "Usage: $0 [vae|ldm|cd|all|all_with_cd]"
		exit 1
		;;
esac

echo "Minimal training done."
