#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

# Required pretrained checkpoint path for EEGPT.
: "${EEGPT_CKPT:?Please set EEGPT_CKPT=/path/to/eegpt_checkpoint.ckpt}"
if [[ ! -f "${EEGPT_CKPT}" ]]; then
  echo "EEGPT checkpoint not found: ${EEGPT_CKPT}"
  exit 1
fi

# Resume LOSO from fold 5 by default.
START_FOLD="${START_FOLD:-5}"
END_FOLD="${END_FOLD:-9}"

# Single-GPU defaults.
NPROC_PER_NODE=1
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-52040}"
RUN_GROUP="${RUN_GROUP:-eegpt_bcic2a_loso_freeze_resume_from5_single}"

if ! [[ "${START_FOLD}" =~ ^[1-9]$ ]] || ! [[ "${END_FOLD}" =~ ^[1-9]$ ]]; then
  echo "START_FOLD and END_FOLD must be in 1..9"
  exit 1
fi
if (( START_FOLD > END_FOLD )); then
  echo "START_FOLD must be <= END_FOLD"
  exit 1
fi

for fold in $(seq "${START_FOLD}" "${END_FOLD}"); do
  echo "============================================================"
  echo "Running LOSO fold ${fold} on single GPU"
  echo "  checkpoint     : ${EEGPT_CKPT}"
  echo "  max_epochs     : ${MAX_EPOCHS}"
  echo "  batch_size     : ${BATCH_SIZE}"
  echo "  num_workers    : ${NUM_WORKERS}"
  echo "  run_group      : ${RUN_GROUP}"
  echo "============================================================"

  EEGPT_CKPT="${EEGPT_CKPT}" \
  FOLD="${fold}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  SEED="${SEED}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  MASTER_PORT_BASE="${MASTER_PORT_BASE}" \
  RUN_GROUP="${RUN_GROUP}" \
  bash scripts/run_eegpt_bcic2a_loso_freeze.sh

done
