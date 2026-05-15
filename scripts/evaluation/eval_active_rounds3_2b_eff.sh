#!/bin/bash
set -e
dataset=${1:-nextqa}
split=${2:-valid}
gpu_id=${3:-1}
bash scripts/evaluation/eval_active_rounds_common_2b_eff.sh "$dataset" "$split" "$gpu_id" 3
