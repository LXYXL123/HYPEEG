#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

if [[ -z "${EEGPT_CKPT:-}" ]]; then
  echo "Please set EEGPT_CKPT to the pretrained EEGPT checkpoint path."
  echo "Example:"
  echo "  EEGPT_CKPT=assets/ckpt/pretrained/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt bash scripts/run_eegpt_bcic2a_loso_freeze.sh"
  exit 1
fi

if [[ ! -f "$EEGPT_CKPT" ]]; then
  echo "EEGPT checkpoint not found: $EEGPT_CKPT"
  exit 1
fi

FOLD="${FOLD:-1}"                  # 1..9 or all
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-52040}"
RUN_GROUP="${RUN_GROUP:-eegpt_bcic2a_loso_freeze}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Clean any stale distributed-launch env from the current shell/tmux session.
unset LOCAL_RANK || true
unset RANK || true
unset WORLD_SIZE || true
unset MASTER_ADDR || true
unset MASTER_PORT || true

run_fold() {
  local fold="$1"
  local port=$((MASTER_PORT_BASE + fold))
  local -a ddp_args=()
  if (( NPROC_PER_NODE > 1 )); then
    ddp_args+=(multi_gpu=true "nproc_per_node=${NPROC_PER_NODE}")
  fi

  echo "============================================================"
  echo "Running EEGPT BCIC-2a LOSO fold ${fold}"
  echo "  checkpoint    : $EEGPT_CKPT"
  echo "  seed          : $SEED"
  echo "  max_epochs    : $MAX_EPOCHS"
  echo "  batch_size    : $BATCH_SIZE"
  echo "  num_workers   : $NUM_WORKERS"
  echo "  master_port   : $port"
  echo "  nproc_per_node: $NPROC_PER_NODE"
  echo "  run_group     : $RUN_GROUP"
  echo "============================================================"

  python3 baseline_main.py \
    conf_file=baseline/eegpt/eegpt_bcic2a_loso_freeze.yaml \
    model_type=eegpt \
    seed="$SEED" \
    master_port="$port" \
    data.batch_size="$BATCH_SIZE" \
    data.num_workers="$NUM_WORKERS" \
    data.datasets.bcic_2a="finetune_loso_fold${fold}" \
    model.pretrained_path="$EEGPT_CKPT" \
    training.max_epochs="$MAX_EPOCHS" \
    logging.run_group="$RUN_GROUP" \
    logging.experiment_name="eegpt_bcic2a_loso_freeze_fold${fold}" \
    logging.tags="[eegpt,bcic_2a,loso,freeze,fold${fold}]" \
    "${ddp_args[@]}"
}

if [[ "$FOLD" == "all" ]]; then
  for fold in $(seq 1 9); do
    run_fold "$fold"
  done
else
  if ! [[ "$FOLD" =~ ^[1-9]$ ]]; then
    echo "Invalid FOLD=$FOLD. Use 1..9 or all."
    exit 1
  fi
  run_fold "$FOLD"
fi
