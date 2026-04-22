#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="/root/EEG-FM-Bench"
if [[ -d "${DEFAULT_REPO_ROOT}" ]]; then
  REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
else
  REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
fi
cd "${REPO_ROOT}"

# Ensure torchrun child processes can import top-level project modules.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"

if command -v torchrun >/dev/null 2>&1; then
  DIST_LAUNCHER=(torchrun)
else
  DIST_LAUNCHER=(python3 -m torch.distributed.run)
fi

"${DIST_LAUNCHER[@]}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/pretrain_relation_cgeom_setup_gap_hf.py \
  --architecture spectral_lite \
  --dataset-configs motor_mv_img:pretrain_bci,cho2017:pretrain_bci \
  --data-root assets/data/pretrain \
  --epochs "${EPOCHS:-200}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --repr-dims "${REPR_DIMS:-128}" \
  --hidden-dims "${HIDDEN_DIMS:-64}" \
  --depth 4 \
  --per-channel-stem-depth "${PER_CHANNEL_STEM_DEPTH:-1}" \
  --spectral-dims "${SPECTRAL_DIMS:-64}" \
  --spectral-win-len 64 \
  --spectral-stride 32 \
  --spectral-freq-low 4 \
  --spectral-freq-high 40 \
  --spectral-mixer-depth "${SPECTRAL_MIXER_DEPTH:-1}" \
  --use-heegnet-lorentz \
  --heegnet-lorentz-dim "${HEEGNET_LORENTZ_DIM:-65}" \
  --lr "${LR:-1e-4}" \
  --weight-decay 1e-2 \
  --max-train-length "${MAX_TRAIN_LENGTH:-512}" \
  --channel-subsample-ratio "${CHANNEL_SUBSAMPLE_RATIO:-0.6}" \
  --min-channel-subsample "${MIN_CHANNEL_SUBSAMPLE:-12}" \
  --lambda-rel "${LAMBDA_REL:-0.1}" \
  --proto-dim "${PROTO_DIM:-256}" \
  --num-prototypes "${NUM_PROTOTYPES:-64}" \
  --proto-teacher-temp "${PROTO_TEACHER_TEMP:-0.04}" \
  --proto-student-temp "${PROTO_STUDENT_TEMP:-0.10}" \
  --proto-center-momentum "${PROTO_CENTER_MOMENTUM:-0.9}" \
  --use-amp \
  --amp-dtype bf16
