#!/bin/bash

REVISION=$(git rev-parse --short HEAD)

uv run src/experiments/cmd/run_gain1.py
G1A_OUT="gain1_ablation_validation_results.txt"
: > $G1A_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_ablation --mode kfold -K5 --run_to_validate "gain1_ablation_baseline_${REVISION}" >> $G1A_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_ablation --mode kfold -K5 --run_to_validate "gain1_ablation_AM_${REVISION}" >> $G1A_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_ablation --mode kfold -K5 --run_to_validate "gain1_ablation_ES_${REVISION}" >> $G1A_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_ablation --mode kfold -K5 --run_to_validate "gain1_ablation_AM_ES_${REVISION}" >> $G1A_OUT

uv run src/experiments/cmd/run_gain1_optim.py
G1O_OUT="gain1_optim_validation_results.txt"
: > $G1O_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_optim --mode kfold -K5 --run_to_validate "gain1_optim_val_auroc_${REVISION}" >> $G1O_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_optim --mode kfold -K5 --run_to_validate "gain1_optim_val_miou_${REVISION}" >> $G1O_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain1_optim --mode kfold -K5 --run_to_validate "gain1_optim_val_map_50_${REVISION}" >> $G1O_OUT

uv run src/experiments/cmd/run_gain2.py
bash src/experiments/cmd/run_gain2_validation.sh "${REVISION}" # gain2_validation_results.txt
uv run src/experiments/cmd/run_gain2_collect.py --experiment_name gain2_layers --revision "${REVISION}" # gain2_layers_ci.csv

G21_OUT="gain2_1_validation_results.txt"
: > $G21_OUT
uv run src/experiments/cmd/run_gain2_1.py
uv run src/experiments/cmd/run_validation.py --experiment_name gain2.1_layers --mode kfold -K5 --run_to_validate "gain2.1_layers_resnet_search_${REVISION}" >> $G21_OUT
uv run src/experiments/cmd/run_validation.py --experiment_name gain2.1_layers --mode kfold -K5 --run_to_validate "gain2.1_layers_resnet_gain_${REVISION}" >> $G21_OUT

uv run src/experiments/cmd/run_gain3.py
G3_OUT="gain3_validation_results.txt"
: > $G3_OUT
# 0-70%, the max bbox ratio the train split supports
for R in $(seq 0 70); do
    uv run src/experiments/cmd/run_validation.py --experiment_name gain3_bbox_percent --mode kfold -K5 --run_to_validate "gain3_bbox_percent_ratio_${R}%_${REVISION}" >> $G3_OUT
done
uv run src/experiments/cmd/run_gain3_collect.py --experiment_name gain3_bbox_percent --revision "${REVISION}" # gain3_runs.csv


uv run src/experiments/cmd/run_gain5.py
bash src/experiments/cmd/run_gain5_validation.sh "${REVISION}" # gain5_validation.txt

uv run src/experiments/cmd/run_gain6.py
uv run src/experiments/cmd/run_gain6_dense.py
G6_OUT="gain6_validation_results.txt"
: > $G6_OUT
# run_gain6.py scale(100) + run_gain6_dense.py filling in 1-30
for W in $(seq 1 30) 40 50 60 70 80 85 90 $(seq 91 100); do
    uv run src/experiments/cmd/run_validation.py --experiment_name gain6_pretrain_percent --mode kfold -K5 --run_to_validate "gain6_pretrain_percent_warmup_${W}%_${REVISION}" >> $G6_OUT
done
uv run src/experiments/cmd/run_gain6_collect.py --experiment_name gain6_pretrain_percent --revision "${REVISION}" # gain6_runs.csv
