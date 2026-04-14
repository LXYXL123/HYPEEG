#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_v2_simple_gate.yaml \
  model_type=relation_cgeom \
  seed=42 \
  master_port=51542 \
  logging.experiment_name=relation_cgeom_bcic2a_v2_simple_gate_seed42 \
  > relation_cgeom_bcic2a_v2_simple_gate_seed42.out 2>&1

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_v2_simple_gate.yaml \
  model_type=relation_cgeom \
  seed=43 \
  master_port=51543 \
  logging.experiment_name=relation_cgeom_bcic2a_v2_simple_gate_seed43 \
  > relation_cgeom_bcic2a_v2_simple_gate_seed43.out 2>&1

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_v2_simple_gate.yaml \
  model_type=relation_cgeom \
  seed=44 \
  master_port=51544 \
  logging.experiment_name=relation_cgeom_bcic2a_v2_simple_gate_seed44 \
  > relation_cgeom_bcic2a_v2_simple_gate_seed44.out 2>&1
