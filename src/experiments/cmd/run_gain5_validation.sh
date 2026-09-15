#!/usr/bin/env bash
REVISION=${1:-84b7abef1b0c}
OUT=gain5_validation.txt
: > $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_L7_${REVISION} --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_L57_${REVISION} --mode kfold -K 5 |& tee -a $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_pp_L7_${REVISION} --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_gradcam_pp_L57_${REVISION} --mode kfold -K 5 |& tee -a $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_layercam_L7_${REVISION} --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain5_cam_variants --run_to_validate gain5_cam_variants_layercam_L57_${REVISION} --mode kfold -K 5 |& tee -a $OUT
