#!/usr/bin/env bash
set -euo pipefail

FLOWAWR_DIR="${FLOWAWR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE_DIR="$(cd "${FLOWAWR_DIR}/.." && pwd)"
DIFFUSIONNFT_DIR="${DIFFUSIONNFT_DIR:-${WORKSPACE_DIR}/DiffusionNFT}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_DIR}/miniconda3}"
CONDA_ENV="DiffusionNFT"
TORCHRUN="${TORCHRUN:-torchrun}"

SD3_MODEL="${SD3_MODEL:-${DIFFUSIONNFT_DIR}/pretrained_models/sd3.5-medium}"
SWANLAB_RUNTIME="${SWANLAB_RUNTIME:-${FLOWAWR_DIR}/.runtime-deps/swanlab-0.9}"
LOG_ROOT="${LOG_ROOT:-${FLOWAWR_DIR}/logs}"
SAVE_DIR="${SAVE_DIR:-${LOG_ROOT}/sd3/pickscore}"
RUN_NAME="${RUN_NAME:-flowawr_sd3_pickscore_dynamic_temperature_eclip_m0p5_p5}"

# Experiment parameters: edit here or set environment variables when launching.
# Explicit --config.* arguments in "$@" override these values.
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
ENERGY_MIN="${ENERGY_MIN:--0.5}"
ENERGY_MAX="${ENERGY_MAX:-5.0}"
TEMP_REFERENCE_START="${TEMP_REFERENCE_START:-100}"
TEMP_REFERENCE_END="${TEMP_REFERENCE_END:-300}"
TEMP_ACTIVATION_STEP="${TEMP_ACTIVATION_STEP:-1000}"
TEMP_WINDOW_SIZE="${TEMP_WINDOW_SIZE:-20}"
TEMP_UPDATE_RATE="${TEMP_UPDATE_RATE:-0.05}"
TEMP_TOLERANCE="${TEMP_TOLERANCE:-0.03}"
TEMP_MIN="${TEMP_MIN:-0.6666666666666666}"

if [[ ! -f "${CONDA_ROOT}/bin/activate" ]]; then
  echo "Conda activation script not found: ${CONDA_ROOT}/bin/activate" >&2
  exit 2
fi
if [[ ! -d "${SD3_MODEL}" ]]; then
  echo "SD3 model directory not found: ${SD3_MODEL}" >&2
  exit 2
fi
if [[ ! -d "${SWANLAB_RUNTIME}/swanlab" ]]; then
  echo "Isolated SwanLab runtime not found: ${SWANLAB_RUNTIME}" >&2
  echo "Install it with:" >&2
  echo "  source ${CONDA_ROOT}/bin/activate ${CONDA_ENV}" >&2
  echo "  python -m pip install --target ${SWANLAB_RUNTIME} swanlab==0.9.0" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}" "${SAVE_DIR}" "${FLOWAWR_DIR}/.cache"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${FLOWAWR_DIR}"

export PYTHONPATH="${SWANLAB_RUNTIME}:${FLOWAWR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTHONUNBUFFERED=1
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

export HF_HOME="${HF_HOME:-${DIFFUSIONNFT_DIR}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${HF_HOME}/diffusers}"

echo "FlowAWR dynamic temperature: 8 GPUs, SD3.5-Medium + PickScore"
echo "model=${SD3_MODEL}"
echo "save_dir=${SAVE_DIR}/${RUN_NAME}/<timestamp>"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_awr_sd3_dynamic_temperature.py \
  --config=config/awr_dynamic_temperature.py:sd3_pickscore_dynamic_temperature \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOG_ROOT}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.train.learning_rate="${LEARNING_RATE}" \
  --config.train.energy_min="${ENERGY_MIN}" \
  --config.train.energy_max="${ENERGY_MAX}" \
  --config.train.dynamic_temperature.reference_start="${TEMP_REFERENCE_START}" \
  --config.train.dynamic_temperature.reference_end="${TEMP_REFERENCE_END}" \
  --config.train.dynamic_temperature.activation_step="${TEMP_ACTIVATION_STEP}" \
  --config.train.dynamic_temperature.window_size="${TEMP_WINDOW_SIZE}" \
  --config.train.dynamic_temperature.update_rate="${TEMP_UPDATE_RATE}" \
  --config.train.dynamic_temperature.tolerance="${TEMP_TOLERANCE}" \
  --config.train.dynamic_temperature.min_temperature="${TEMP_MIN}" \
  "$@"
