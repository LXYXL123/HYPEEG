#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

declare -a DDP_ARGS=()
if (( NPROC_PER_NODE > 1 )); then
  DDP_ARGS+=(multi_gpu=true "nproc_per_node=${NPROC_PER_NODE}")
fi

if [[ -z "${EEGPT_CKPT:-}" ]]; then
  echo "Please set EEGPT_CKPT to the pretrained EEGPT checkpoint path."
  echo "Example:"
  echo "  EEGPT_CKPT=assets/ckpt/pretrained/eegpt/eegpt_pretrained.ckpt bash scripts/run_eegpt_bcic2a_pretrained_fullfinetune.sh"
  exit 1
fi

if [[ ! -f "$EEGPT_CKPT" ]]; then
  echo "EEGPT checkpoint not found: $EEGPT_CKPT"
  exit 1
fi

echo "Using EEGPT pretrained checkpoint: $EEGPT_CKPT"
echo "Full fine-tuning: training.freeze_encoder=false"
echo "nproc_per_node: $NPROC_PER_NODE"

python3 baseline_main.py \
  conf_file=baseline/eegpt/eegpt_bcic2a.yaml \
  model_type=eegpt \
  training.freeze_encoder=false \
  model.pretrained_path="$EEGPT_CKPT" \
  logging.run_group=eegpt_bcic2a_pretrained_fullfinetune \
  logging.experiment_name=eegpt_bcic2a_pretrained_fullfinetune \
  logging.tags='[eegpt,bcic_2a,pretrained,fullfinetune]' \
  "${DDP_ARGS[@]}"
