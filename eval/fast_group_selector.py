import json, os, re, csv, copy
from collections import Counter, defaultdict

RAW = "outputs_2b_active/nextqa_valid_full_sweep_r5_raw/output.json"
OUT = "outputs_2b_active/r5_fast_group_selector"
os.makedirs(OUT, exist_ok=True)

LETTERS = "ABCDE"

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
                return LETTERS[i]
            if opt.lower() in resp.lower():
                return LETTERS[i]
    return None

def uniq_rounds(rounds):
    out, seen = [], set()
    for r in rounds:
        k = r.get("candidate_id") or str(r.get("span")) or str(r.get("round"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def make_keys(task, qtype, cand_round, r1_pred, cand_pred):
    return {
        "task_round": (task, cand_round),
        "task_round_new": (task, cand_round, cand_pred),
        "task_transition": (task, r1_pred, cand_pred),
        "task_round_transition": (task, cand_round, r1_pred, cand_pred),
        "round_transition": (cand_round, r1_pred, cand_pred),
        "qtype_round_transition": (qtype, cand_round, r1_pred, cand_pred),
        "task_qtype_round_transition": (task, qtype, cand_round, r1_pred, cand_pred),
    }

def apply_choice(sample, chosen):
    t = copy.deepcopy(sample)
    if chosen is None:
        return t
    t["selected_round"] = chosen.get("round", 1)
    t["accepted_round"] = chosen.get("round", t.get("accepted_round"))
    t["response"] = chosen.get("response", t.get("response"))
    t["answerer_response"] = chosen.get("response", t.get("answerer_response"))
    t["final_answer_text"] = chosen.get("answer_text", t.get("final_answer_text"))
    t["reverse_question"] = chosen.get("reverse_question")
    t["faithfulness_score"] = chosen.get("faithfulness_score")
    t["sufficiency_score"] = chosen.get("sufficiency_score")
    t["selected_span"] = chosen.get("span", t.get("selected_span"))
    t["selected_evidence_state"] = chosen.get("evidence_state", t.get("selected_evidence_state"))
    t["selected_candidate_id"] = chosen.get("candidate_id", t.get("selected_candidate_id"))
    t["selected_candidate_kind"] = chosen.get("candidate_kind", t.get("selected_candidate_kind"))
    return t

print("loading raw output...")
data = json.load(open(RAW, "r", encoding="utf-8"))
print("samples:", len(data))

samples = []
records_by_keytype = defaultdict(list)

for idx, s in enumerate(data):
    rounds = uniq_rounds(s.get("evidence_rounds") or [])
    options = s.get("options")
    gt = s.get("ans")
    task = s.get("task", "UNK")
    q = s.get("question") or s.get("query") or ""
    qtype = q.strip().split()[0].lower() if q.strip() else "unk"

    if rounds:
        r1 = rounds[0]
        r1_pred = pred_letter(r1.get("response"), r1.get("answer_text"), options)
    else:
        r1 = None
        r1_pred = pred_letter(s.get("response"), s.get("final_answer_text"), options)

    old_correct = (r1_pred == gt)

    samples.append({
        "idx": idx,
        "gt": gt,
        "task": task,
        "qtype": qtype,
        "r1": r1,
        "r1_pred": r1_pred,
        "old_correct": old_correct,
    })

    if not rounds or not r1_pred:
        continue

    for r in rounds[1:]:
        cand_pred = pred_letter(r.get("response"), r.get("answer_text"), options)
        if not cand_pred or cand_pred == r1_pred:
            continue

        cand_round = r.get("round", 1)
        new_correct = (cand_pred == gt)
        delta = int(new_correct) - int(old_correct)

        keys = make_keys(task, qtype, cand_round, r1_pred, cand_pred)

        for kt, key in keys.items():
            records_by_keytype[kt].append({
                "idx": idx,
                "key": key,
                "round": cand_round,
                "pred": cand_pred,
                "delta": delta,
                "old_correct": old_correct,
                "new_correct": new_correct,
                "chosen": r,
            })

def evaluate(key_type, min_support, min_net, min_precision):
    records = records_by_keytype[key_type]

    stats = defaultdict(lambda: {"n": 0, "gain": 0, "fix": 0, "hurt": 0})
    for rec in records:
        st = stats[rec["key"]]
        st["n"] += 1
        st["gain"] += rec["delta"]
        if rec["delta"] > 0:
            st["fix"] += 1
        elif rec["delta"] < 0:
            st["hurt"] += 1

    good_keys = set()
    for k, st in stats.items():
        n = st["n"]
        fix = st["fix"]
        hurt = st["hurt"]
        precision = fix / max(1, fix + hurt)
        if n >= min_support and st["gain"] >= min_net and precision >= min_precision:
            good_keys.add(k)

    choices = {}
    for rec in records:
        if rec["key"] not in good_keys:
            continue
        st = stats[rec["key"]]
        precision = st["fix"] / max(1, st["fix"] + st["hurt"])
        score = (st["gain"], precision, -rec["round"])
        idx = rec["idx"]
        if idx not in choices or score > choices[idx][0]:
            choices[idx] = (score, rec)

    correct = 0
    changed = 0
    fix = 0
    hurt = 0
    selected_rounds = Counter()

    for meta in samples:
        idx = meta["idx"]
        gt = meta["gt"]
        r1_pred = meta["r1_pred"]
        old_correct = meta["old_correct"]

        if idx in choices:
            rec = choices[idx][1]
            final_pred = rec["pred"]
            final_round = rec["round"]
            new_correct = rec["new_correct"]
        else:
            final_pred = r1_pred
            final_round = 1
            new_correct = old_correct

        if final_pred == gt:
            correct += 1

        if final_pred != r1_pred:
            changed += 1
            if not old_correct and new_correct:
                fix += 1
            elif old_correct and not new_correct:
                hurt += 1

        selected_rounds[str(final_round)] += 1

    total = len(samples)
    return {
        "key_type": key_type,
        "min_support": min_support,
        "min_net": min_net,
        "min_precision": min_precision,
        "samples": total,
        "correct": correct,
        "acc": correct / total * 100,
        "changed": changed,
        "r1_wrong_to_final_right": fix,
        "r1_right_to_final_wrong": hurt,
        "net_gain": fix - hurt,
        "num_good_keys": len(good_keys),
        "selected_rounds": dict(selected_rounds),
        "choices": choices,
        "good_keys": good_keys,
        "stats": stats,
    }

key_types = list(records_by_keytype.keys())
min_supports = [1, 2, 3, 5, 8, 10]
min_nets = [1, 2, 3, 5]
min_precisions = [0.50, 0.55, 0.60, 0.65, 0.70]

rows = []
best = None

print("searching selectors...")
for kt in key_types:
    for ms in min_supports:
        for mn in min_nets:
            for mp in min_precisions:
                res = evaluate(kt, ms, mn, mp)
                row = {k: v for k, v in res.items() if k not in ("choices", "good_keys", "stats")}
                rows.append(row)
                if best is None or res["acc"] > best["acc"]:
                    best = res

rows.sort(key=lambda x: x["acc"], reverse=True)

leaderboard = os.path.join(OUT, "leaderboard.csv")
with open(leaderboard, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "key_type", "min_support", "min_net", "min_precision",
        "samples", "correct", "acc", "changed",
        "r1_wrong_to_final_right", "r1_right_to_final_wrong",
        "net_gain", "num_good_keys", "selected_rounds"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("Top selectors:")
for r in rows[:15]:
    print(r)

print("\nBEST:")
print({k: v for k, v in best.items() if k not in ("choices", "good_keys", "stats")})

best_dir = os.path.join(OUT, "best_output")
os.makedirs(best_dir, exist_ok=True)

out_data = []
for idx, s in enumerate(data):
    if idx in best["choices"]:
        chosen = best["choices"][idx][1]["chosen"]
    else:
        rounds = uniq_rounds(s.get("evidence_rounds") or [])
        chosen = rounds[0] if rounds else None
    out_data.append(apply_choice(s, chosen))

json.dump(out_data, open(os.path.join(best_dir, "output.json"), "w", encoding="utf-8"), ensure_ascii=False)

groups_path = os.path.join(OUT, "best_good_groups.csv")
with open(groups_path, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "n", "gain", "fix", "hurt"])
    for k in best["good_keys"]:
        st = best["stats"][k]
        w.writerow([k, st["n"], st["gain"], st["fix"], st["hurt"]])

print("saved leaderboard:", leaderboard)
print("saved best output:", os.path.join(best_dir, "output.json"))
print("saved groups:", groups_path)

