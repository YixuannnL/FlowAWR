#!/usr/bin/env bash
set -euo pipefail

# Defaults to both sides. Select either single-side variant with, for example:
# bash scripts/launchers/run_flowawr_sd3_pickscore_rarity_redistribution_8gpu.sh \
#   --config=config/awr_rarity_redistribution.py:sd3_pickscore_positive
# Strength overrides use --config.rarity.positive_strength=0.5 (or negative).
CONFIG_PATH="config/awr_rarity_redistribution.py:sd3_pickscore_both"
FORWARDED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config=*|-config=*) CONFIG_PATH="${1#*=}"; shift ;;
    --config|-config)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "--config requires a config file and entry point" >&2
        exit 2
      fi
      CONFIG_PATH="$2"; shift 2 ;;
    *) FORWARDED_ARGS+=("$1"); shift ;;
  esac
done
case "${CONFIG_PATH##*:}" in
  sd3_pickscore_positive) VARIANT=positive ;;
  sd3_pickscore_negative) VARIANT=negative ;;
  sd3_pickscore_both) VARIANT=both ;;
  *) VARIANT=custom ;;
esac

FLOWAWR_DIR="${FLOWAWR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE_DIR="$(cd "${FLOWAWR_DIR}/.." && pwd)"
DIFFUSIONNFT_DIR="${DIFFUSIONNFT_DIR:-${WORKSPACE_DIR}/DiffusionNFT}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_DIR}/miniconda3}"
CONDA_ENV="DiffusionNFT"
TORCHRUN="${TORCHRUN:-torchrun}"

SD3_MODEL="${SD3_MODEL:-${DIFFUSIONNFT_DIR}/pretrained_models/sd3.5-medium}"
SWANLAB_RUNTIME="${SWANLAB_RUNTIME:-${FLOWAWR_DIR}/.runtime-deps/swanlab-0.9}"
LOG_ROOT="${LOG_ROOT:-${FLOWAWR_DIR}/logs}"
SAVE_DIR="${SAVE_DIR:-${LOG_ROOT}/sd3/pickscore_rarity_redistribution_${VARIANT}}"
RUN_NAME="${RUN_NAME:-flowawr_sd3_pickscore_rarity_redistribution_${VARIANT}}"

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

# Match the existing rarity launcher environment and cache configuration.
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

echo "FlowAWR rarity redistribution (${VARIANT}): 8 GPUs, SD3.5-Medium + PickScore"
echo "model=${SD3_MODEL}"
echo "save_dir=${SAVE_DIR}/${RUN_NAME}/<timestamp>"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_awr_sd3_rarity_redistribution.py \
  --config="${CONFIG_PATH}" \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOG_ROOT}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  ${FORWARDED_ARGS[@]+"${FORWARDED_ARGS[@]}"}
