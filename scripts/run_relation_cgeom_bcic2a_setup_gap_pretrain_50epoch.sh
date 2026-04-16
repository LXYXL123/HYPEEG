#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAIN_ROOT="${PRETRAIN_ROOT:-assets/run/pretrain/relation_cgeom}"
PRETRAIN_PATTERN="${PRETRAIN_PATTERN:-setup_gap_disc_motor_mv_img-pretrain_bci_cho2017-pretrain_bci}"
CKPT_NAME="${CKPT_NAME:-relation_cgeom_encoder_best.pt}"

LATEST_CKPT="$(
  find "$PRETRAIN_ROOT" \
    -path "*${PRETRAIN_PATTERN}*" \
    -name "$CKPT_NAME" \
    -printf "%T@ %p\n" \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-
)"

if [[ -z "$LATEST_CKPT" ]]; then
  echo "No checkpoint found under $PRETRAIN_ROOT"
  echo "Pattern: *${PRETRAIN_PATTERN}*"
  echo "Checkpoint name: $CKPT_NAME"
  echo "Expected something like:"
  echo "  assets/run/pretrain/relation_cgeom/${PRETRAIN_PATTERN}/local_*/$CKPT_NAME"
  exit 1
fi

echo "Using pretrained checkpoint: $LATEST_CKPT"

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_setup_conditioned.yaml \
  model_type=relation_cgeom \
  training.max_epochs=50 \
  training.grad_accum_steps=4 \
  training.use_amp=true \
  data.batch_size=8 \
  data.num_workers=8 \
  model.pretrained_path="$LATEST_CKPT" \
  logging.run_group=bcic2a_setup_gap_disc_pretrain_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_setup_gap_disc_pretrain_50epoch \
  logging.tags='[relation_cgeom,bcic_2a,setup_gap_disc_pretrain,50epoch]'
