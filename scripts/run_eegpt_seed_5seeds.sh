#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EEGPT_CKPT="${EEGPT_CKPT:-assets/ckpt/pretrained/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt}"
DATASET_CONFIG="${DATASET_CONFIG:-finetune}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
RUN_DIR="${RUN_DIR:-assets/run2}"
RUN_GROUP="${RUN_GROUP:-eegpt_seed_${DATASET_CONFIG}_freeze_${FREEZE_ENCODER}_5seeds}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-52240}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RANDOM_SEEDS="${RANDOM_SEEDS:-42 43 44 45 46}"

if [[ -z "$EEGPT_CKPT" ]]; then
  echo "Please set EEGPT_CKPT to the pretrained EEGPT checkpoint path."
  exit 1
fi

if [[ ! -f "$EEGPT_CKPT" ]]; then
  echo "EEGPT checkpoint not found: $EEGPT_CKPT"
  echo "Set it explicitly, for example:"
  echo "  EEGPT_CKPT=assets/ckpt/pretrained/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt bash scripts/run_eegpt_seed_5seeds.sh"
  exit 1
fi

# Clean stale torchrun env from tmux/current shell.
unset LOCAL_RANK || true
unset RANK || true
unset WORLD_SIZE || true
unset MASTER_ADDR || true
unset MASTER_PORT || true

declare -a DDP_ARGS=()
if (( NPROC_PER_NODE > 1 )); then
  DDP_ARGS+=(multi_gpu=true "nproc_per_node=${NPROC_PER_NODE}")
fi

idx=0
for random_seed in $RANDOM_SEEDS; do
  port=$((MASTER_PORT_BASE + idx))
  experiment_name="eegpt_seed_${DATASET_CONFIG}_seed${random_seed}_freeze_${FREEZE_ENCODER}_${MAX_EPOCHS}epoch"
  tags="[eegpt,seed,${DATASET_CONFIG},random_seed${random_seed},freeze_${FREEZE_ENCODER},${MAX_EPOCHS}epoch]"

  echo "============================================================"
  echo "Running EEGPT on SEED"
  echo "  dataset_config : $DATASET_CONFIG"
  echo "  random_seed    : $random_seed"
  echo "  checkpoint     : $EEGPT_CKPT"
  echo "  freeze_encoder : $FREEZE_ENCODER"
  echo "  max_epochs     : $MAX_EPOCHS"
  echo "  batch_size     : $BATCH_SIZE"
  echo "  num_workers    : $NUM_WORKERS"
  echo "  nproc_per_node : $NPROC_PER_NODE"
  echo "  master_port    : $port"
  echo "  run_group      : $RUN_GROUP"
  echo "============================================================"

  python3 baseline_main.py \
    conf_file=baseline/eegpt/eegpt_seed.yaml \
    model_type=eegpt \
    seed="$random_seed" \
    master_port="$port" \
    data.datasets.seed="$DATASET_CONFIG" \
    data.batch_size="$BATCH_SIZE" \
    data.num_workers="$NUM_WORKERS" \
    model.pretrained_path="$EEGPT_CKPT" \
    training.max_epochs="$MAX_EPOCHS" \
    training.freeze_encoder="$FREEZE_ENCODER" \
    logging.run_dir="$RUN_DIR" \
    logging.run_group="$RUN_GROUP" \
    logging.experiment_name="$experiment_name" \
    logging.tags="$tags" \
    "${DDP_ARGS[@]}"

  idx=$((idx + 1))
done
