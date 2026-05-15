import json, re, os, csv
from collections import Counter, defaultdict

OUT = "outputs_2b_active/method_support_tables"
os.makedirs(OUT, exist_ok=True)

paths = {
    "baseline": "outputs_2b_active/nextqa_valid_feedback12_stopgate/output.json",
    "raw_r5": "outputs_2b_active/nextqa_valid_full_sweep_r5_raw/output.json",
    "raw_vote": "outputs_2b_active/nextqa_valid_full_sweep_r5_vote/output.json",
    "dedup_vote": "outputs_2b_active/nextqa_valid_full_sweep_r5_selective_dedup/output.json",
    "group": "outputs_2b_active/r5_fast_group_selector/best_output/output.json",
    "cv": "outputs_2b_active/r5_cv_group_selector/output.json",
    "oracle": "outputs_2b_active/nextqa_valid_sweep_oracle_merged/output.json",
}

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def pred_letter(resp, ans_text='', options=None):
    resp = resp or ''
    ans_text = ans_text or ''
    m = re.search(r'\b([A-E])\)', resp)
    if m:
        return m.group(1)
    m = re.search(r'^\s*([A-E])\b', resp)
    if m:
        return m.group(1)
    if options:
        for i, opt in enumerate(options):
            opt = str(opt).strip()
            if ans_text.strip().lower() == opt.lower():
                return "ABCDE"[i]
            if opt.lower() in resp.lower():
                return "ABCDE"[i]
    return None

def get_pred(s):
    return pred_letter(s.get("response"), s.get("final_answer_text"), s.get("options"))

def qtype(q):
    q = (q or "").strip().lower()
    if q.startswith("why"):
        return "why"
    if q.startswith("how many"):
        return "how_many"
    if q.startswith("how"):
        return "how"
    if q.startswith("what"):
        return "what"
    if q.startswith("where"):
        return "where"
    if q.startswith("when"):
        return "when"
    if "after" in q or "before" in q or "then" in q:
        return "temporal"
    return "other"

def correct(s):
    return get_pred(s) == s.get("ans")

def round_pred(r, options=None):
    return pred_letter(r.get("response"), r.get("answer_text"), options)

data = {k: load(v) for k, v in paths.items() if os.path.exists(v)}
base = data["baseline"]
base_by_uid = {s["uid"]: s for s in base}

# 1. Main summary
main_rows = []
for name, samples in data.items():
    total = len(samples)
    corr = sum(correct(s) for s in samples)
    changed = 0
    w2r = 0
    r2w = 0
    for s in samples:
        uid = s["uid"]
        if uid not in base_by_uid:
            continue
        bp = get_pred(base_by_uid[uid])
        sp = get_pred(s)
        gt = s.get("ans")
        if bp != sp:
            changed += 1
        if bp != gt and sp == gt:
            w2r += 1
        if bp == gt and sp != gt:
            r2w += 1
    main_rows.append({
        "run": name,
        "samples": total,
        "correct": corr,
        "acc": round(corr / total * 100, 2) if total else 0,
        "changed": changed,
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net_gain": w2r-r2w,
    })

# 2. Round contribution from raw R5
round_rows = []
if "raw_r5" in data:
    for s in data["raw_r5"]:
        gt = s.get("ans")
        options = s.get("options")
        rounds = s.get("evidence_rounds") or []
        seen = set()
        for r in rounds:
            rp = round_pred(r, options)
            rr = r.get("round")
            key = (rr, rp)
            if key in seen:
                continue
            seen.add(key)
            round_rows.append({
                "uid": s.get("uid"),
                "task": s.get("task"),
                "qtype": qtype(s.get("question") or s.get("query")),
                "round": rr,
                "pred": rp,
                "gt": gt,
                "correct": int(rp == gt),
                "candidate_id": r.get("candidate_id"),
                "candidate_kind": r.get("candidate_kind"),
                "sufficiency_score": r.get("sufficiency_score"),
                "faithfulness_score": r.get("faithfulness_score"),
                "verifier_score": r.get("verifier_score"),
                "action": r.get("action"),
            })

# 3. qtype results for group
qtype_counter = defaultdict(lambda: Counter())
for s in data.get("group", []):
    qt = qtype(s.get("question") or s.get("query"))
    qtype_counter[qt]["total"] += 1
    qtype_counter[qt]["correct"] += int(correct(s))
    b = base_by_uid.get(s["uid"])
    if b:
        bp, sp, gt = get_pred(b), get_pred(s), s.get("ans")
        qtype_counter[qt]["changed"] += int(bp != sp)
        qtype_counter[qt]["w2r"] += int(bp != gt and sp == gt)
        qtype_counter[qt]["r2w"] += int(bp == gt and sp != gt)

# 4. candidate kind / selected round stats
kind_counter = Counter()
round_counter = Counter()
cid_counter = Counter()
for s in data.get("group", []):
    kind_counter[str(s.get("selected_candidate_kind"))] += 1
    round_counter[str(s.get("selected_round"))] += 1
    cid_counter[str(s.get("selected_candidate_id"))] += 1

# 5. score calibration from raw R5 rounds
score_bins = defaultdict(lambda: Counter())
def bin_score(x):
    if x is None:
        return "None"
    try:
        x = float(x)
    except Exception:
        return "None"
    lo = int(x * 10) / 10
    hi = lo + 0.1
    if lo >= 1.0:
        return "[1.0]"
    return f"[{lo:.1f},{hi:.1f})"

for row in round_rows:
    b = bin_score(row["sufficiency_score"])
    score_bins[b]["total"] += 1
    score_bins[b]["correct"] += row["correct"]

# Write CSVs
with open(os.path.join(OUT, "main_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
    w.writeheader()
    w.writerows(main_rows)

with open(os.path.join(OUT, "round_contribution.csv"), "w", newline="", encoding="utf-8") as f:
    if round_rows:
        w = csv.DictWriter(f, fieldnames=list(round_rows[0].keys()))
        w.writeheader()
        w.writerows(round_rows)

with open(os.path.join(OUT, "qtype_results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qtype", "total", "correct", "acc", "changed", "wrong_to_right", "right_to_wrong", "net_gain"])
    for qt, c in sorted(qtype_counter.items()):
        total = c["total"]
        correct_n = c["correct"]
        w.writerow([qt, total, correct_n, round(correct_n/total*100,2) if total else 0,
                    c["changed"], c["w2r"], c["r2w"], c["w2r"]-c["r2w"]])

with open(os.path.join(OUT, "candidate_kind_stats.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["type", "name", "count"])
    for k,v in kind_counter.items():
        w.writerow(["candidate_kind", k, v])
    for k,v in round_counter.items():
        w.writerow(["selected_round", k, v])
    for k,v in cid_counter.items():
        w.writerow(["candidate_id", k, v])

with open(os.path.join(OUT, "score_calibration.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sufficiency_bin", "total_rounds", "correct_rounds", "round_acc"])
    for b,c in sorted(score_bins.items()):
        total = c["total"]
        corr = c["correct"]
        w.writerow([b, total, corr, round(corr/total*100,2) if total else 0])

# Markdown summary
md = []
md.append("# Method Support Tables\n")

md.append("## Main Summary\n")
md.append("| Run | Samples | Correct | Acc | Changed | Wrong→Right | Right→Wrong | Net Gain |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in main_rows:
    md.append(f"| {r['run']} | {r['samples']} | {r['correct']} | {r['acc']} | {r['changed']} | {r['wrong_to_right']} | {r['right_to_wrong']} | {r['net_gain']} |")

md.append("\n## Q-type Results for Group Selector\n")
md.append("| Q-type | Total | Correct | Acc | Changed | Wrong→Right | Right→Wrong | Net Gain |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for qt, c in sorted(qtype_counter.items()):
    total = c["total"]
    corr = c["correct"]
    md.append(f"| {qt} | {total} | {corr} | {round(corr/total*100,2) if total else 0} | {c['changed']} | {c['w2r']} | {c['r2w']} | {c['w2r']-c['r2w']} |")

md.append("\n## Selected Candidate Kind / Round\n")
md.append("| Type | Name | Count |")
md.append("|---|---|---:|")
for k,v in kind_counter.items():
    md.append(f"| candidate_kind | {k} | {v} |")
for k,v in round_counter.items():
    md.append(f"| selected_round | {k} | {v} |")
for k,v in cid_counter.items():
    md.append(f"| candidate_id | {k} | {v} |")

md.append("\n## Sufficiency Score Calibration\n")
md.append("| Sufficiency Bin | Total Rounds | Correct Rounds | Round Acc |")
md.append("|---|---:|---:|---:|")
for b,c in sorted(score_bins.items()):
    total = c["total"]
    corr = c["correct"]
    md.append(f"| {b} | {total} | {corr} | {round(corr/total*100,2) if total else 0} |")

with open(os.path.join(OUT, "method_support_tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(open(os.path.join(OUT, "method_support_tables.md"), "r", encoding="utf-8").read())
print("\nSaved to:", OUT)
