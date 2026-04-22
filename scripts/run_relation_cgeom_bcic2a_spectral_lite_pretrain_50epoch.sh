#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAIN_ROOT="${PRETRAIN_ROOT:-assets/run/pretrain/relation_cgeom}"
PRETRAIN_PATTERN="${PRETRAIN_PATTERN:-proto_distill_spectral_lite_motor_mv_img-pretrain_bci_cho2017-pretrain_bci}"
CKPT_NAME="${CKPT_NAME:-relation_cgeom_encoder_last.pt}"
FOLD="${FOLD:-1}"
DATASET_CONFIG="${DATASET_CONFIG:-finetune_loso_fold${FOLD}}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

declare -a DDP_ARGS=()
if (( NPROC_PER_NODE > 1 )); then
  DDP_ARGS+=(multi_gpu=true "nproc_per_node=${NPROC_PER_NODE}")
fi

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

RUN_GROUP="bcic2a_loso_fold${FOLD}_spectral_lite_proto_pretrain_${MAX_EPOCHS}epoch"
EXPERIMENT_NAME="relation_cgeom_bcic2a_loso_fold${FOLD}_spectral_lite_proto_pretrain_${MAX_EPOCHS}epoch"
TAGS="[relation_cgeom,bcic_2a,loso,fold${FOLD},spectral_lite,proto_pretrain,${MAX_EPOCHS}epoch]"

if [[ "$FREEZE_ENCODER" == "true" ]]; then
  RUN_GROUP="bcic2a_loso_fold${FOLD}_spectral_lite_proto_pretrain_freeze_${MAX_EPOCHS}epoch"
  EXPERIMENT_NAME="relation_cgeom_bcic2a_loso_fold${FOLD}_spectral_lite_proto_pretrain_freeze_${MAX_EPOCHS}epoch"
  TAGS="[relation_cgeom,bcic_2a,loso,fold${FOLD},spectral_lite,proto_pretrain,freeze_encoder,${MAX_EPOCHS}epoch]"
fi

echo "Using pretrained checkpoint: $LATEST_CKPT"
echo "Dataset: bcic_2a/$DATASET_CONFIG"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Grad accumulation steps: $GRAD_ACCUM_STEPS"
echo "Freeze encoder: $FREEZE_ENCODER"
echo "nproc_per_node: $NPROC_PER_NODE"

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_spectral_lite.yaml \
  model_type=relation_cgeom \
  training.max_epochs="$MAX_EPOCHS" \
  training.freeze_encoder="$FREEZE_ENCODER" \
  training.grad_accum_steps="$GRAD_ACCUM_STEPS" \
  training.use_amp=true \
  data.batch_size="$BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  data.datasets.bcic_2a="$DATASET_CONFIG" \
  model.pretrained_path="$LATEST_CKPT" \
  logging.run_group="$RUN_GROUP" \
  logging.experiment_name="$EXPERIMENT_NAME" \
  logging.tags="$TAGS" \
  "${DDP_ARGS[@]}"
