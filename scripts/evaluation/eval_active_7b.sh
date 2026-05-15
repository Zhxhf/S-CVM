#!/bin/bash

set -e

# Usage:
#   bash scripts/evaluation/eval_active_7b.sh <dataset> [split] [variant] [gpu_devices]
# Example:
#   bash scripts/evaluation/eval_active_7b.sh nextqa valid full 1

dataset=$1
split=${2:-"test"}
variant=${3:-"full"}
gpu_devices_arg=${4:-""}

if [ -n "$gpu_devices_arg" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu_devices_arg"
fi
# Safe default: use a single GPU unless the user explicitly passes more.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

model_gnd_path="model_zoo/VideoMind-7B"
model_ver_path="model_zoo/VideoMind-7B"
model_pla_path="model_zoo/VideoMind-7B"
model_ans_path="model_zoo/VideoMind-7B"

pred_path="outputs_7b_active/${dataset}_${split}_${variant}"

echo -e "\e[1;36mEvaluating active-evidence variant:\e[0m $dataset ($split) variant=$variant"
echo -e "\e[1;33mUsing GPU devices:\e[0m $CUDA_VISIBLE_DEVICES"

IFS="," read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto.py \
        --dataset $dataset \
        --split $split \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --model_ver_path $model_ver_path \
        --model_pla_path $model_pla_path \
        --model_ans_path $model_ans_path \
        --active_variant $variant \
        --sufficiency_threshold 0.48 \
        --sufficiency_beta 0.25 \
        --max_rounds 2 \
        --candidate_pool_size 1 \
        --boundary_expand_ratio 0.08 \
        --acceptance_margin 0.12 \
        --verifier_accept_threshold 0.48 \
        --verifier_skip_reverse_threshold 0.46 \
        --min_feedback_gap 0.16 \
        --feedback_verifier_floor 0.42 \
        --sufficiency_eval_rounds 1 \
        --save_round_details \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

python videomind/eval/eval_auto.py $pred_path --dataset $dataset
python videomind/eval/summarize_active.py $pred_path
