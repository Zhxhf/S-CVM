import argparse
import csv
import json
import math
import os
import re
from statistics import mean


def load_output_json(pred_path: str):
    candidates = [
        pred_path,
        os.path.join(pred_path, "output.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Cannot find output json under: {pred_path}")


def read_wall_clock_seconds(eff_dir: str):
    p = os.path.join(eff_dir, "time.txt")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    m = re.search(r"wall_clock_seconds=(\d+)", txt)
    return int(m.group(1)) if m else None


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def summarize_gpu_csv(eff_dir: str):
    p = os.path.join(eff_dir, "gpu.csv")
    if not os.path.isfile(p):
        return {}

    gpu_util = []
    mem_util = []
    mem_used = []
    mem_total = []
    power = []

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            gpu_util.append(safe_float(row[2]))
            mem_util.append(safe_float(row[3]))
            mem_used.append(safe_float(row[4]))
            mem_total.append(safe_float(row[5]))
            power.append(safe_float(row[6]))

    out = {}
    if gpu_util:
        out["gpu_util_mean"] = mean(gpu_util)
        out["gpu_util_max"] = max(gpu_util)
    if mem_util:
        out["gpu_mem_util_mean"] = mean(mem_util)
        out["gpu_mem_util_max"] = max(mem_util)
    if mem_used:
        out["gpu_mem_used_mb_mean"] = mean(mem_used)
        out["gpu_mem_used_mb_max"] = max(mem_used)
    if mem_total:
        out["gpu_mem_total_mb"] = max(mem_total)
    if power:
        out["gpu_power_w_mean"] = mean(power)
        out["gpu_power_w_max"] = max(power)
    return out


def summarize_proc_csv(eff_dir: str):
    p = os.path.join(eff_dir, "proc.csv")
    if not os.path.isfile(p):
        return {}

    cpu = []
    mem = []
    rss = []
    vsz = []
    read_b = []
    write_b = []

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 9:
                continue
            cpu.append(safe_float(row[3]))
            mem.append(safe_float(row[4]))
            rss.append(safe_float(row[5]))
            vsz.append(safe_float(row[6]))
            read_b.append(safe_float(row[7]))
            write_b.append(safe_float(row[8]))

    out = {}
    if cpu:
        out["cpu_pct_mean"] = mean(cpu)
        out["cpu_pct_max"] = max(cpu)
    if mem:
        out["proc_mem_pct_mean"] = mean(mem)
        out["proc_mem_pct_max"] = max(mem)
    if rss:
        out["proc_rss_kb_mean"] = mean(rss)
        out["proc_rss_kb_max"] = max(rss)
    if vsz:
        out["proc_vsz_kb_mean"] = mean(vsz)
        out["proc_vsz_kb_max"] = max(vsz)
    if read_b:
        out["proc_read_bytes_max"] = max(read_b)
    if write_b:
        out["proc_write_bytes_max"] = max(write_b)
    return out


def count_text_tokens_est(text):
    if not text:
        return 0
    return len(re.findall(r"\S+", str(text)))


def summarize_outputs(data):
    if isinstance(data, dict):
        data = list(data.values())

    n = len(data)
    executed_rounds = []
    selected_rounds = []
    accepted_rounds = []
    faithfulness = []
    text_tokens = 0

    for item in data:
        er = item.get("executed_rounds", None)
        if er is None:
            rounds = item.get("evidence_rounds", [])
            er = len(rounds) if isinstance(rounds, list) and rounds else 1
        executed_rounds.append(er)

        sr = item.get("selected_round", None)
        if sr is not None:
            selected_rounds.append(sr)

        ar = item.get("accepted_round", None)
        if ar is not None:
            accepted_rounds.append(ar)

        fs = item.get("faithfulness_score", item.get("overlap", None))
        if fs is not None:
            try:
                faithfulness.append(float(fs))
            except Exception:
                pass

        text_tokens += count_text_tokens_est(item.get("final_answer_text", item.get("answer", "")))
        text_tokens += count_text_tokens_est(item.get("reverse_question", ""))

        for r in item.get("evidence_rounds", []):
            text_tokens += count_text_tokens_est(r.get("answer_text", ""))
            text_tokens += count_text_tokens_est(r.get("reverse_question", ""))

    out = {
        "processed_samples": n,
        "avg_executed_rounds": mean(executed_rounds) if executed_rounds else 0.0,
        "sum_executed_rounds": sum(executed_rounds) if executed_rounds else 0,
        "text_tokens_est_total": text_tokens,
    }
    if selected_rounds:
        out["avg_selected_round"] = mean(selected_rounds)
    if accepted_rounds:
        out["avg_accepted_round"] = mean(accepted_rounds)
    if faithfulness:
        out["avg_faithfulness_score"] = mean(faithfulness)
    return out


def count_model_params(model_path: str):
    from videomind.model.builder import build_model
    model, _ = build_model(model_path, device="cpu")
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    return {
        "total_params": total_params,
        "trainable_params_current_flag": trainable_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument("--eff_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    data = load_output_json(args.pred_path)
    out_stats = summarize_outputs(data)
    gpu_stats = summarize_gpu_csv(args.eff_dir)
    proc_stats = summarize_proc_csv(args.eff_dir)
    wall = read_wall_clock_seconds(args.eff_dir)

    merged = {}
    merged.update(out_stats)
    merged.update(gpu_stats)
    merged.update(proc_stats)
    merged["wall_clock_seconds"] = wall

    if wall and wall > 0:
        merged["samples_per_second"] = out_stats["processed_samples"] / wall
        merged["clips_per_second_effective"] = out_stats["sum_executed_rounds"] / wall
        merged["text_tokens_est_per_second"] = out_stats["text_tokens_est_total"] / wall

    if args.model_path:
        try:
            merged.update(count_model_params(args.model_path))
        except Exception as e:
            merged["model_param_count_error"] = str(e)

    print(json.dumps(merged, indent=2, ensure_ascii=False))

    out_json = os.path.join(args.eff_dir, "efficiency_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    out_csv = os.path.join(args.eff_dir, "efficiency_summary.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in merged.items():
            writer.writerow([k, v])


if __name__ == "__main__":
    main()
