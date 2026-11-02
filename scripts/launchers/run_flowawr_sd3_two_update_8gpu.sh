#!/usr/bin/env bash
set -euo pipefail

FLOWAWR_DIR="${FLOWAWR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_ROOT="${VENV_ROOT:-/local-ssd/flowawr-venv}"
TORCHRUN="${TORCHRUN:-${VENV_ROOT}/bin/torchrun}"

SD3_MODEL="${SD3_MODEL:-/cpfs/public_models/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80}"
SWANLAB_RUNTIME="${SWANLAB_RUNTIME:-/local-ssd/flowawr-swanlab}"
# Supported existing presets; no config file is modified.
REWARD="${REWARD:-pickscore}"
case "${REWARD}" in
  pickscore|hpsv2|ocr|geneval) ;;
  *) echo "REWARD must be pickscore, hpsv2, ocr, or geneval" >&2; exit 2 ;;
esac
LOG_ROOT="${LOG_ROOT:-${KOALA_RUN_DIR:?KOALA_RUN_DIR must point to this run on CPFS}/logs}"
SAVE_DIR="${SAVE_DIR:-${LOG_ROOT}/sd3/two_update/${REWARD}}"
RUN_NAME="${RUN_NAME:-flowawr_sd3_two_update_${REWARD}_a6_1}"

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "Run scripts/launchers/prepare_flowawr_h20_runtime.sh first; missing ${VENV_ROOT}/bin/activate" >&2
  exit 2
fi
if [[ ! -d "${SD3_MODEL}" ]]; then
  echo "SD3 model directory not found: ${SD3_MODEL}" >&2
  exit 2
fi
if [[ ! -d "${SWANLAB_RUNTIME}/swanlab" ]]; then
  echo "Isolated SwanLab runtime not found: ${SWANLAB_RUNTIME}" >&2
  echo "Install it with:" >&2
  echo "  source ${VENV_ROOT}/bin/activate" >&2
  echo "  python -m pip install --target ${SWANLAB_RUNTIME} swanlab==0.9.0" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}" "${SAVE_DIR}" "${FLOWAWR_DIR}/.cache"

source "${VENV_ROOT}/bin/activate"
cd "${FLOWAWR_DIR}"

# Isolate SwanLab's protobuf 6 from the pinned wandb/protobuf 5 environment.
export PYTHONPATH="${SWANLAB_RUNTIME}:${FLOWAWR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTHONUNBUFFERED=1
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

# Fixed CPFS assets; disposable caches stay on the node-local disk.
export REWARD_CKPTS_DIR="${REWARD_CKPTS_DIR:-/cpfs/yuxuanyao/models/flowawr/reward_ckpts_20261101}"
export GENEVAL_OPENCLIP_PATH="${GENEVAL_OPENCLIP_PATH:-${REWARD_CKPTS_DIR}/geneval/openclip/ViT-L-14-state_dict.pt}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/local-ssd/flowawr-hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${HF_HOME}/diffusers}"

echo "FlowAWR frozen-current target: 8 GPUs, SD3.5-Medium + ${REWARD}, two full-batch updates"
echo "model=${SD3_MODEL}"
echo "reward_ckpts=${REWARD_CKPTS_DIR}"
PREFLIGHT_ARGS=()
if [[ "${REWARD}" == pickscore ]]; then PREFLIGHT_ARGS+=(--pickscore); fi
python scripts/preflight_flowawr_sd3.py --model "${SD3_MODEL}" --reward "${REWARD_CKPTS_DIR}" "${PREFLIGHT_ARGS[@]}"
echo "save_dir=${SAVE_DIR}/${RUN_NAME}/<timestamp>"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_awr_sd3_two_update.py \
  --config="config/awr.py:sd3_${REWARD}" \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOG_ROOT}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.train.ema=False \
  --config.train.beta=0.0 \
  --config.train.num_inner_epochs=2 \
  --config.decay_type=0 \
  "$@"
