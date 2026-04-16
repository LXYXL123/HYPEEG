#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/pretrain_relation_cgeom_setup_gap_hf.py \
  --dataset-configs motor_mv_img:pretrain_bci,cho2017:pretrain_bci \
  --data-root assets/data/pretrain \
  --array-cache-dir assets/data/pretrain/array_cache \
  --epochs 10 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --grad-accum-steps 4 \
  --num-workers 4 \
  --max-train-length 512 \
  --channel-subsample-ratio 0.6 \
  --min-channel-subsample 12 \
  --depth 7 \
  --hidden-dims 64 \
  --repr-dims 320 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --warmup-ratio 0.05 \
  --grad-clip 1.0 \
  --ema-momentum 0.995 \
  --lambda-disc-max 0.05 \
  --disc-tau 0.2 \
  --disc-queue-size 512 \
  --use-amp \
  --amp-dtype bf16 \
  --use-setup-conditioned \
  --use-variable-channel-frontend
