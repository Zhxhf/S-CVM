#!/bin/bash
set -e

dataset=$1
split=$2
gpu_id=$3
max_rounds=$4

if [ -z "$dataset" ] || [ -z "$split" ] || [ -z "$gpu_id" ] || [ -z "$max_rounds" ]; then
  echo "Usage: bash scripts/evaluation/eval_active_rounds_common_2b_eff.sh <dataset> <split> <gpu> <rounds>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONPATH="./:$PYTHONPATH"
unset HF_ENDPOINT
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

variant="full"
round_tag="r${max_rounds}"

model_path="/home/ubuntu/videomind/VideoMind/model_zoo/VideoMind-2B"

pred_path="outputs_2b_active/${dataset}_${split}_${variant}_${round_tag}"
eff_dir="logs/efficiency/${dataset}_${split}_${variant}_${round_tag}_gpu${gpu_id}"

mkdir -p "$pred_path"
mkdir -p "$eff_dir"

echo "Evaluating $dataset ($split) variant=$variant rounds=$max_rounds on physical GPU $gpu_id"
echo "pred_path=$pred_path"
echo "eff_dir=$eff_dir"

SECONDS=0

(
  echo "timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
  while true; do
    nvidia-smi -i "$gpu_id" \
      --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits
    sleep 2
  done
) > "${eff_dir}/gpu.csv" &
GPU_MON_PID=$!

python videomind/eval/infer_auto.py \
  --dataset "$dataset" \
  --split "$split" \
  --pred_path "$pred_path" \
  --model_gnd_path "$model_path" \
  --model_ver_path "$model_path" \
  --model_pla_path "$model_path" \
  --model_ans_path "$model_path" \
  --active_variant "$variant" \
  --sufficiency_threshold 0.48 \
  --sufficiency_beta 0.25 \
  --candidate_pool_size 1 \
  --boundary_expand_ratio 0.08 \
  --acceptance_margin 0.12 \
  --verifier_accept_threshold 0.48 \
  --verifier_skip_reverse_threshold 0.46 \
  --min_feedback_gap 0.16 \
  --feedback_verifier_floor 0.42 \
  --sufficiency_eval_rounds 1 \
  --max_rounds "$max_rounds" \
  --selection_rule best_faithfulness \
  --save_round_details \
  --planner_max_pixels $((24 * 28 * 28)) \
  --planner_max_frames 24 \
  --planner_fps 0.5 \
  --grounder_max_pixels $((48 * 28 * 28)) \
  --grounder_max_frames 48 \
  --grounder_fps 0.5 \
  --verifier_topk 3 \
  --verifier_max_pixels $((48 * 28 * 28)) \
  --verifier_max_frames 24 \
  --verifier_fps 1.0 \
  > "${eff_dir}/run.log" 2>&1 &
MAIN_PID=$!

(
  echo "timestamp,pid,etime,%cpu,%mem,rss,vsz,read_bytes,write_bytes"
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    TS=$(date "+%F %T")
    PS_LINE=$(ps -p "$MAIN_PID" -o pid=,etime=,%cpu=,%mem=,rss=,vsz=)
    if [ -n "$PS_LINE" ]; then
      RB=$(grep '^read_bytes:' /proc/$MAIN_PID/io 2>/dev/null | awk '{print $2}')
      WB=$(grep '^write_bytes:' /proc/$MAIN_PID/io 2>/dev/null | awk '{print $2}')
      echo "$TS,$PS_LINE,${RB:-0},${WB:-0}"
    fi
    sleep 2
  done
) > "${eff_dir}/proc.csv" &
PROC_MON_PID=$!

wait "$MAIN_PID"
STATUS=$?

kill "$GPU_MON_PID" 2>/dev/null || true
kill "$PROC_MON_PID" 2>/dev/null || true

ELAPSED=$SECONDS
echo "wall_clock_seconds=${ELAPSED}" | tee "${eff_dir}/time.txt"

python videomind/eval/eval_auto.py "$pred_path" --dataset "$dataset" | tee "${eff_dir}/eval.log"
python videomind/eval/summarize_active.py "$pred_path" | tee "${eff_dir}/summary.log"
python videomind/eval/summarize_round_efficiency.py "$pred_path" --eff_dir "$eff_dir" --model_path "$model_path" | tee "${eff_dir}/efficiency.log"

exit $STATUS
