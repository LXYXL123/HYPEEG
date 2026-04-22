#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --master_port="${MASTER_PORT:-51620}" \
  scripts/pretrain_relation_cgeom_setup_gap_hf.py \
  --architecture spectral_lite \
  --dataset-configs motor_mv_img:pretrain_bci,cho2017:pretrain_bci \
  --data-root assets/data/pretrain \
  --epochs "${EPOCHS:-10}" \
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
  --proto-teacher-temp "${PROTO_TEACHER_TEMP:-0.10}" \
  --proto-teacher-temp-start "${PROTO_TEACHER_TEMP_START:-0.20}" \
  --proto-teacher-temp-warmup-ratio "${PROTO_TEACHER_TEMP_WARMUP_RATIO:-0.30}" \
  --proto-student-temp "${PROTO_STUDENT_TEMP:-0.10}" \
  --proto-center-momentum "${PROTO_CENTER_MOMENTUM:-0.9}" \
  --lambda-proto-balance "${LAMBDA_PROTO_BALANCE:-0.1}" \
  --use-amp \
  --amp-dtype bf16
