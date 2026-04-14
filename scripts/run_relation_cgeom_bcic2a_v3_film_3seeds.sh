#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_v3_film.yaml \
  model_type=relation_cgeom \
  seed=42 \
  master_port=51642 \
  logging.experiment_name=relation_cgeom_bcic2a_v3_film_seed42

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_v3_film.yaml \
  model_type=relation_cgeom \
  seed=43 \
  master_port=51643 \
  logging.experiment_name=relation_cgeom_bcic2a_v3_film_seed43

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a_v3_film.yaml \
  model_type=relation_cgeom \
  seed=44 \
  master_port=51644 \
  logging.experiment_name=relation_cgeom_bcic2a_v3_film_seed44
