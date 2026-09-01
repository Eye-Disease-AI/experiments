#!/usr/bin/env bash
OUT=gain5_validation.txt
: > $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_L7_84b7abef1b0c --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_L57_84b7abef1b0c --mode kfold -K 5 |& tee -a $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_pp_L7_84b7abef1b0c --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_pp_L57_84b7abef1b0c --mode kfold -K 5 |& tee -a $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_layercam_L7_84b7abef1b0c --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_layercam_L57_84b7abef1b0c --mode kfold -K 5 |& tee -a $OUT
