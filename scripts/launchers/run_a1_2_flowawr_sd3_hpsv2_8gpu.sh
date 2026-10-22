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
SAVE_DIR="${SAVE_DIR:-${LOG_ROOT}/sd3/hpsv2}"
RUN_NAME="${RUN_NAME:-flowawr_sd3_hpsv2}"

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

echo "A1.2 FlowAWR: 8 GPUs, SD3.5-Medium + HPSv2.1"
echo "model=${SD3_MODEL}"
echo "save_dir=${SAVE_DIR}/${RUN_NAME}/<timestamp>"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_awr_sd3.py \
  --config=config/awr.py:sd3_hpsv2 \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOG_ROOT}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  "$@"
