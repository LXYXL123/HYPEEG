#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FOLD="${FOLD:-1}"
DATASET_CONFIG="${DATASET_CONFIG:-finetune_loso_fold${FOLD}}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-51892}"
RUN_DIR="${RUN_DIR:-assets/run2}"
SEED="${SEED:-42}"

RUN_GROUP="bcic2a_loso_fold${FOLD}_spectral_lite_scratch_full_${MAX_EPOCHS}epoch"
EXPERIMENT_NAME="relation_cgeom_bcic2a_loso_fold${FOLD}_spectral_lite_scratch_full_${MAX_EPOCHS}epoch"
TAGS="[relation_cgeom,bcic_2a,loso,fold${FOLD},spectral_lite,scratch,full_finetune,${MAX_EPOCHS}epoch]"

echo "Scratch full-parameter Relation-CGeom spectral-lite downstream run"
echo "Dataset: bcic_2a/$DATASET_CONFIG"
echo "Max epochs: $MAX_EPOCHS"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Grad accumulation steps: $GRAD_ACCUM_STEPS"
echo "nproc_per_node: $NPROC_PER_NODE"
echo "Global batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))"
echo "Seed: $SEED"
echo "Run group: $RUN_GROUP"

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_spectral_lite.yaml \
  model_type=relation_cgeom \
  seed="$SEED" \
  master_port="$MASTER_PORT" \
  multi_gpu=true \
  nproc_per_node="$NPROC_PER_NODE" \
  training.max_epochs="$MAX_EPOCHS" \
  training.freeze_encoder=false \
  training.grad_accum_steps="$GRAD_ACCUM_STEPS" \
  training.use_amp=true \
  data.batch_size="$BATCH_SIZE" \
  data.num_workers="$NUM_WORKERS" \
  data.datasets.bcic_2a="$DATASET_CONFIG" \
  model.pretrained_path=null \
  logging.run_dir="$RUN_DIR" \
  logging.run_group="$RUN_GROUP" \
  logging.experiment_name="$EXPERIMENT_NAME" \
  logging.tags="$TAGS"
