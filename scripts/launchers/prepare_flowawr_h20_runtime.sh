#!/usr/bin/env bash
# Restore the CPU-validated runtime on node-local storage. Does not train.
set -euo pipefail
RUNTIME_ASSET="${RUNTIME_ASSET:-/cpfs/yuxuanyao/models/flowawr/runtime_20261101}"
cd "${RUNTIME_ASSET}"
ARCHIVE_HASH="6b46bfe8e0d7a36ecf4faa4888b11e5accb513691e8b587dc7f94d036dfca753"
printf '%s  runtime.tar.gz\n' "${ARCHIVE_HASH}" | sha256sum --check -
if [[ ! -f /local-ssd/flowawr-runtime.ready || "$(cat /local-ssd/flowawr-runtime.ready)" != "${ARCHIVE_HASH}" ]]; then
  if [[ -e /local-ssd/flowawr-venv || -e /local-ssd/flowawr-swanlab ]]; then
    echo 'Runtime directories already exist without a ready marker; inspect before restoring.' >&2
    exit 2
  fi
  tar -xzf runtime.tar.gz -C /local-ssd
  /local-ssd/flowawr-venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12); import torch; assert torch.__version__ == "2.6.0+cu124"'
  printf '%s\n' "${ARCHIVE_HASH}" > /local-ssd/flowawr-runtime.ready
fi
