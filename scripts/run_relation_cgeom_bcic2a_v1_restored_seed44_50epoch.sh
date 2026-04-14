#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=44 \
  master_port=51844 \
  training.max_epochs=50 \
  logging.run_group=v1_restored_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_v1_restored_seed44_50epoch
