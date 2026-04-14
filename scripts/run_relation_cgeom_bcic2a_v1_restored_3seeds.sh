#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=42 \
  master_port=51742 \
  logging.run_group=v1_restored \
  logging.experiment_name=relation_cgeom_bcic2a_v1_restored_seed42

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=43 \
  master_port=51743 \
  logging.run_group=v1_restored \
  logging.experiment_name=relation_cgeom_bcic2a_v1_restored_seed43

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=44 \
  master_port=51744 \
  logging.run_group=v1_restored \
  logging.experiment_name=relation_cgeom_bcic2a_v1_restored_seed44
