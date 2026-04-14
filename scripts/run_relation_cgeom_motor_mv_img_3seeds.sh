#!/usr/bin/env bash
set -euo pipefail

cd /root/EEG-FM-Bench

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_motor_mv_img.yaml \
  model_type=relation_cgeom \
  seed=42 \
  master_port=51942 \
  training.max_epochs=50 \
  logging.run_group=motor_mv_img \
  logging.experiment_name=relation_cgeom_motor_mv_img_seed42_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_motor_mv_img.yaml \
  model_type=relation_cgeom \
  seed=43 \
  master_port=51943 \
  training.max_epochs=50 \
  logging.run_group=motor_mv_img \
  logging.experiment_name=relation_cgeom_motor_mv_img_seed43_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_motor_mv_img.yaml \
  model_type=relation_cgeom \
  seed=44 \
  master_port=51944 \
  training.max_epochs=50 \
  logging.run_group=motor_mv_img \
  logging.experiment_name=relation_cgeom_motor_mv_img_seed44_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_motor_mv_img.yaml \
  model_type=relation_cgeom \
  seed=45 \
  master_port=51945 \
  training.max_epochs=50 \
  logging.run_group=motor_mv_img \
  logging.experiment_name=relation_cgeom_motor_mv_img_seed45_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_motor_mv_img.yaml \
  model_type=relation_cgeom \
  seed=46 \
  master_port=51946 \
  training.max_epochs=50 \
  logging.run_group=motor_mv_img \
  logging.experiment_name=relation_cgeom_motor_mv_img_seed46_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=42 \
  master_port=52042 \
  training.max_epochs=50 \
  logging.run_group=bcic2a_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_seed42_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=43 \
  master_port=52043 \
  training.max_epochs=50 \
  logging.run_group=bcic2a_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_seed43_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=44 \
  master_port=52044 \
  training.max_epochs=50 \
  logging.run_group=bcic2a_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_seed44_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=45 \
  master_port=52045 \
  training.max_epochs=50 \
  logging.run_group=bcic2a_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_seed45_50epoch

python3 baseline_main.py \
  conf_file=baseline/relation_cgeom/relation_cgeom_bcic2a.yaml \
  model_type=relation_cgeom \
  seed=46 \
  master_port=52046 \
  training.max_epochs=50 \
  logging.run_group=bcic2a_50epoch \
  logging.experiment_name=relation_cgeom_bcic2a_seed46_50epoch
