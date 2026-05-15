#!/bin/bash

set -e

# Usage:
#   bash scripts/evaluation/eval_active_ablation_2b.sh <dataset> [split] [gpu_devices]
# Example:
#   bash scripts/evaluation/eval_active_ablation_2b.sh nextqa valid 1

dataset=$1
split=${2:-"test"}
gpu_devices_arg=${3:-""}

for variant in baseline es_only es_fa es_fa_ea full; do
    bash scripts/evaluation/eval_active_2b.sh $dataset $split $variant $gpu_devices_arg
done
