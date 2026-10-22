#!/usr/bin/env bash
set -euo pipefail

FLOWAWR_DIR="${FLOWAWR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE_DIR="$(cd "${FLOWAWR_DIR}/.." && pwd)"
DIFFUSIONNFT_DIR="${DIFFUSIONNFT_DIR:-${WORKSPACE_DIR}/DiffusionNFT}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_DIR}/miniconda3}"
CONDA_ENV="DiffusionNFT"
TORCHRUN="${TORCHRUN:-torchrun}"

SD3_MODEL="${SD3_MODEL:-${DIFFUSIONNFT_DIR}/pretrained_models/sd3.5-medium}"
LORA_PATH="${LORA_PATH:-}"
OUT_DIR="${OUT_DIR:-${FLOWAWR_DIR}/probe_runs/flowawr_probe_ckpt900_8gpu}"

if [[ ! -f "${CONDA_ROOT}/bin/activate" ]]; then
  echo "Conda activation script not found: ${CONDA_ROOT}/bin/activate" >&2
  exit 2
fi
if [[ ! -d "${SD3_MODEL}" ]]; then
  echo "SD3 model directory not found: ${SD3_MODEL}" >&2
  exit 2
fi
if [[ -z "${LORA_PATH}" ]]; then
  echo "Set LORA_PATH to checkpoint-900/lora before launching P1." >&2
  exit 2
fi
if [[ ! -f "${LORA_PATH}/adapter_config.json" ]]; then
  echo "Invalid LoRA directory (adapter_config.json missing): ${LORA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${LORA_PATH}/adapter_model.safetensors" && ! -f "${LORA_PATH}/adapter_model.bin" ]]; then
  echo "Invalid LoRA directory (adapter weights missing): ${LORA_PATH}" >&2
  exit 2
fi
if [[ -d "${OUT_DIR}" && -n "$(find "${OUT_DIR}" -mindepth 1 -print -quit)" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "Output directory is not empty: ${OUT_DIR}" >&2
  echo "Choose another OUT_DIR or set ALLOW_OVERWRITE=1." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${FLOWAWR_DIR}/.cache"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${FLOWAWR_DIR}"

export PYTHONPATH="${FLOWAWR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${DIFFUSIONNFT_DIR}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${HF_HOME}/diffusers}"

# Eight GPUs preserve the CSV experiment's 1152 samples per round via
# 8 ranks x 9 samples per batch x 16 batches.
echo "P1 gradient-SNR probe: 8 GPUs, checkpoint-900 LoRA"
echo "lora_path=${LORA_PATH}"
echo "out_dir=${OUT_DIR}"

"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/probe_grad_snr_sd3.py \
  --config=config/awr.py:sd3_pickscore \
  --config.pretrained.model="${SD3_MODEL}" \
  --lora_path="${LORA_PATH}" \
  --out_dir="${OUT_DIR}" \
  --num_rounds=2 \
  --samples_per_round=1152 \
  --sample_epoch=900 \
  --sketch_dim=65536 \
  --sketch_seed=0 \
  --noise_seed=1234 \
  --loss_scale=1024 \
  --exact_check=true \
  "$@"
