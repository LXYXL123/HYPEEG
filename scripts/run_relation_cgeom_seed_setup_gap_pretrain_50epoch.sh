#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAIN_ROOT="${PRETRAIN_ROOT:-assets/run/pretrain/relation_cgeom}"
PRETRAIN_PATTERN="${PRETRAIN_PATTERN:-setup_gap_disc_motor_mv_img-pretrain_bci_cho2017-pretrain_bci}"
CKPT_NAME="${CKPT_NAME:-relation_cgeom_encoder_best.pt}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"

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

RUN_GROUP="seed_setup_gap_disc_pretrain_${MAX_EPOCHS}epoch"
EXPERIMENT_NAME="relation_cgeom_seed_setup_gap_disc_pretrain_${MAX_EPOCHS}epoch"
TAGS="[relation_cgeom,seed,setup_gap_disc_pretrain,${MAX_EPOCHS}epoch]"

if [[ "$FREEZE_ENCODER" == "true" ]]; then
  RUN_GROUP="seed_setup_gap_disc_pretrain_freeze_${MAX_EPOCHS}epoch"
  EXPERIMENT_NAME="relation_cgeom_seed_setup_gap_disc_pretrain_freeze_${MAX_EPOCHS}epoch"
  TAGS="[relation_cgeom,seed,setup_gap_disc_pretrain,freeze_encoder,${MAX_EPOCHS}epoch]"
fi

echo "Using pretrained checkpoint: $LATEST_CKPT"
echo "Dataset: seed/finetune"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Grad accumulation steps: $GRAD_ACCUM_STEPS"
echo "Freeze encoder: $FREEZE_ENCODER"

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_seed.yaml \
  model_type=relation_cgeom \
  training.max_epochs="$MAX_EPOCHS" \
  training.freeze_encoder="$FREEZE_ENCODER" \
  training.grad_accum_steps="$GRAD_ACCUM_STEPS" \
  training.use_amp=true \
  data.batch_size="$BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  model.pretrained_path="$LATEST_CKPT" \
  model.use_setup_conditioned=true \
  model.use_variable_channel_frontend=true \
  model.per_channel_stem_depth=2 \
  model.channel_attn_heads=4 \
  logging.run_group="$RUN_GROUP" \
  logging.experiment_name="$EXPERIMENT_NAME" \
  logging.tags="$TAGS"
