#!/usr/bin/env bash
OUT=validation_results.txt
: > $OUT

# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L1_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L3_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L5_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L7_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L13_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L15_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L17_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L35_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L37_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L57_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L135_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L137_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L157_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L357_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
# uv run src/experiments/cmd/run_validation.py --experiment_name gain2_layers --run_to_validate gain2_layers_L1357_9f39adb04fbd --mode kfold -K 5 |& tee -a $OUT
