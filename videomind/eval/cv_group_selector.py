import json, os, re, csv, copy, hashlib
from collections import Counter, defaultdict

RAW = "outputs_2b_active/nextqa_valid_full_sweep_r5_raw/output.json"
OUT = "outputs_2b_active/r5_cv_group_selector"
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

def fold_id(sample, k=5):
    uid = str(sample.get("uid", ""))
    h = hashlib.md5(uid.encode("utf-8")).hexdigest()
    return int(h, 16) % k

def group_key(sample, r1_pred, cand_pred, cand_round):
    task = sample.get("task", "UNK")
    q = sample.get("question") or sample.get("query") or ""
    qtype = q.strip().split()[0].lower() if q.strip() else "unk"
    return (task, qtype, cand_round, r1_pred, cand_pred)

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

print("loading raw R5 output...")
data = json.load(open(RAW, "r", encoding="utf-8"))
print("samples:", len(data))

all_records = []
meta = []

for idx, s in enumerate(data):
    rounds = uniq_rounds(s.get("evidence_rounds") or [])
    options = s.get("options")
    gt = s.get("ans")

    if rounds:
        r1 = rounds[0]
        r1_pred = pred_letter(r1.get("response"), r1.get("answer_text"), options)
    else:
        r1 = None
        r1_pred = pred_letter(s.get("response"), s.get("final_answer_text"), options)

    old_correct = (r1_pred == gt)

    meta.append({
        "idx": idx,
        "fold": fold_id(s),
        "gt": gt,
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

        all_records.append({
            "idx": idx,
            "fold": fold_id(s),
            "key": group_key(s, r1_pred, cand_pred, cand_round),
            "round": cand_round,
            "pred": cand_pred,
            "chosen": r,
            "delta": delta,
            "old_correct": old_correct,
            "new_correct": new_correct,
        })

def learn_good_keys(train_folds, min_support=1, min_net=1, min_precision=0.50):
    stats = defaultdict(lambda: {"n": 0, "gain": 0, "fix": 0, "hurt": 0})

    for rec in all_records:
        if rec["fold"] not in train_folds:
            continue
        st = stats[rec["key"]]
        st["n"] += 1
        st["gain"] += rec["delta"]
        if rec["delta"] > 0:
            st["fix"] += 1
        elif rec["delta"] < 0:
            st["hurt"] += 1

    good = set()
    for k, st in stats.items():
        precision = st["fix"] / max(1, st["fix"] + st["hurt"])
        if st["n"] >= min_support and st["gain"] >= min_net and precision >= min_precision:
            good.add(k)

    return good, stats

out_data = [None] * len(data)
fold_rows = []
global_selected = Counter()
global_correct = 0
global_changed = 0
global_fix = 0
global_hurt = 0

for test_fold in range(5):
    train_folds = [f for f in range(5) if f != test_fold]
    good_keys, stats = learn_good_keys(train_folds)

    fold_correct = 0
    fold_total = 0
    fold_changed = 0
    fold_fix = 0
    fold_hurt = 0
    fold_selected = Counter()

    records_by_idx = defaultdict(list)
    for rec in all_records:
        if rec["fold"] != test_fold:
            continue
        if rec["key"] not in good_keys:
            continue

        st = stats[rec["key"]]
        precision = st["fix"] / max(1, st["fix"] + st["hurt"])
        score = (st["gain"], precision, -rec["round"])
        records_by_idx[rec["idx"]].append((score, rec))

    for m in meta:
        if m["fold"] != test_fold:
            continue

        idx = m["idx"]
        s = data[idx]
        gt = m["gt"]
        r1_pred = m["r1_pred"]
        old_correct = m["old_correct"]

        chosen = m["r1"]

        if idx in records_by_idx:
            chosen = sorted(records_by_idx[idx], key=lambda x: x[0], reverse=True)[0][1]["chosen"]

        if chosen is not None:
            final_pred = pred_letter(chosen.get("response"), chosen.get("answer_text"), s.get("options"))
            selected_round = chosen.get("round", 1)
        else:
            final_pred = pred_letter(s.get("response"), s.get("final_answer_text"), s.get("options"))
            selected_round = 1

        new_correct = (final_pred == gt)

        fold_total += 1
        global_selected[str(selected_round)] += 1
        fold_selected[str(selected_round)] += 1

        if new_correct:
            fold_correct += 1
            global_correct += 1

        if final_pred != r1_pred:
            fold_changed += 1
            global_changed += 1
            if not old_correct and new_correct:
                fold_fix += 1
                global_fix += 1
            elif old_correct and not new_correct:
                fold_hurt += 1
                global_hurt += 1

        out_data[idx] = apply_choice(s, chosen)

    fold_rows.append({
        "fold": test_fold,
        "samples": fold_total,
        "correct": fold_correct,
        "acc": round(fold_correct / fold_total * 100, 2),
        "changed": fold_changed,
        "wrong_to_right": fold_fix,
        "right_to_wrong": fold_hurt,
        "net_gain": fold_fix - fold_hurt,
        "num_good_keys": len(good_keys),
        "selected_rounds": dict(fold_selected),
    })

summary = {
    "samples": len(data),
    "correct": global_correct,
    "acc": round(global_correct / len(data) * 100, 2),
    "changed": global_changed,
    "wrong_to_right": global_fix,
    "right_to_wrong": global_hurt,
    "net_gain": global_fix - global_hurt,
    "selected_rounds": dict(global_selected),
    "folds": fold_rows,
}

json.dump(out_data, open(os.path.join(OUT, "output.json"), "w", encoding="utf-8"), ensure_ascii=False)
json.dump(summary, open(os.path.join(OUT, "cv_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

with open(os.path.join(OUT, "fold_results.csv"), "w", encoding="utf-8", newline="") as f:
    fieldnames = ["fold", "samples", "correct", "acc", "changed", "wrong_to_right", "right_to_wrong", "net_gain", "num_good_keys", "selected_rounds"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(fold_rows)

print("CV summary:")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("saved:", os.path.join(OUT, "output.json"))
print("saved:", os.path.join(OUT, "cv_summary.json"))
print("saved:", os.path.join(OUT, "fold_results.csv"))