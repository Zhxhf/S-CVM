#!/bin/bash
set -e

dataset=${1:-nextqa}
split=${2:-valid}
gpu_id=1

export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONPATH="./:$PYTHONPATH"
unset HF_ENDPOINT
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

variant="full"
round_tag="r1"
model_path="/home/ubuntu/videomind/VideoMind/model_zoo/VideoMind-2B"

pred_path="outputs_2b_active/${dataset}_${split}_${variant}_${round_tag}"
eff_dir="logs/efficiency/${dataset}_${split}_${variant}_${round_tag}_gpu${gpu_id}"

mkdir -p "$pred_path"
mkdir -p "$eff_dir"

echo "Evaluating $dataset ($split) variant=$variant rounds=1 on physical GPU $gpu_id"
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
  --max_rounds 1 \
  --selection_rule best_faithfulness \
  --save_round_details \
  --planner_max_pixels 28224 \
  --planner_max_frames 24 \
  --planner_fps 0.5 \
  --grounder_max_pixels 28224 \
  --grounder_max_frames 32 \
  --grounder_fps 0.5 \
  --verifier_topk 3 \
  --verifier_max_pixels 28224 \
  --verifier_max_frames 16 \
  --verifier_fps 0.75 \
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
