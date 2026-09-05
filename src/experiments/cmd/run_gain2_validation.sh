#!/usr/bin/env bash
REVISION=${1:-9f39adb04fbd}
OUT=gain2_validation_results.txt
: > $OUT

uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L1_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L3_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L5_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L7_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L13_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L15_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L17_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L35_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L37_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L57_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L135_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L137_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L157_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L357_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate "gain2_layers_L1357_${REVISION}" --mode kfold -K 5 |& tee -a $OUT
