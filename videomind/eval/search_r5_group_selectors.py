import json, os, re, csv, copy
from collections import Counter, defaultdict

RAW = "outputs_2b_active/nextqa_valid_full_sweep_r5_raw/output.json"
OUT_DIR = "outputs_2b_active/r5_group_selector_search"
os.makedirs(OUT_DIR, exist_ok=True)

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
    out = []
    seen = set()
    for r in rounds:
        k = r.get("candidate_id") or str(r.get("span")) or str(r.get("round"))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

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

def group_key(sample, r1_pred, cand_pred, cand_round, key_type):
    task = sample.get("task", "UNK")
    q = sample.get("question") or sample.get("query") or ""
    qtype = q.strip().split()[0].lower() if q.strip() else "unk"

    if key_type == "task_round":
        return (task, cand_round)
    if key_type == "task_round_new":
        return (task, cand_round, cand_pred)
    if key_type == "task_transition":
        return (task, r1_pred, cand_pred)
    if key_type == "task_round_transition":
        return (task, cand_round, r1_pred, cand_pred)
    if key_type == "qtype_round_transition":
        return (qtype, cand_round, r1_pred, cand_pred)
    if key_type == "task_qtype_round_transition":
        return (task, qtype, cand_round, r1_pred, cand_pred)
    if key_type == "round_transition":
        return (cand_round, r1_pred, cand_pred)
    return (task, cand_round)

def collect_candidates(data, key_type):
    records = []

    for idx, s in enumerate(data):
        rounds = uniq_rounds(s.get("evidence_rounds") or [])
        if not rounds:
            continue

        options = s.get("options")
        gt = s.get("ans")

        r1 = rounds[0]
        r1_pred = pred_letter(r1.get("response"), r1.get("answer_text"), options)
        if not r1_pred:
            continue

        old_correct = (r1_pred == gt)

        for r in rounds[1:]:
            cand_pred = pred_letter(r.get("response"), r.get("answer_text"), options)
            if not cand_pred or cand_pred == r1_pred:
                continue

            new_correct = (cand_pred == gt)
            delta = int(new_correct) - int(old_correct)
            cand_round = r.get("round", 1)

            key = group_key(s, r1_pred, cand_pred, cand_round, key_type)

            records.append({
                "idx": idx,
                "key": key,
                "round": cand_round,
                "delta": delta,
                "old_correct": old_correct,
                "new_correct": new_correct,
                "cand": r,
                "r1_pred": r1_pred,
                "cand_pred": cand_pred,
            })

    return records

def run_policy(data, key_type, min_support, min_net, min_precision):
    records = collect_candidates(data, key_type)

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

    by_idx = defaultdict(list)
    for rec in records:
        if rec["key"] in good_keys:
            st = stats[rec["key"]]
            precision = st["fix"] / max(1, st["fix"] + st["hurt"])
            score = (st["gain"], precision, -rec["round"])
            by_idx[rec["idx"]].append((score, rec))

    output = []
    correct = 0
    changed = 0
    fix = 0
    hurt = 0
    selected_rounds = Counter()

    for idx, s in enumerate(data):
        rounds = uniq_rounds(s.get("evidence_rounds") or [])
        options = s.get("options")
        gt = s.get("ans")

        r1 = rounds[0] if rounds else None
        r1_pred = pred_letter(r1.get("response"), r1.get("answer_text"), options) if r1 else pred_letter(s.get("response"), s.get("final_answer_text"), options)

        chosen = r1
        if idx in by_idx:
            chosen = sorted(by_idx[idx], key=lambda x: x[0], reverse=True)[0][1]["cand"]

        final_pred = pred_letter(chosen.get("response"), chosen.get("answer_text"), options) if chosen else pred_letter(s.get("response"), s.get("final_answer_text"), options)

        if final_pred == gt:
            correct += 1

        if final_pred != r1_pred:
            changed += 1
            if r1_pred != gt and final_pred == gt:
                fix += 1
            elif r1_pred == gt and final_pred != gt:
                hurt += 1

        selected_rounds[str(chosen.get("round", 1) if chosen else 1)] += 1
        t = apply_choice(s, chosen)
        t["group_selector_key_type"] = key_type
        output.append(t)

    total = len(data)
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
        "num_good_groups": len(good_keys),
        "selected_rounds": dict(selected_rounds),
        "output": output,
        "stats": stats,
        "good_keys": good_keys,
    }

def main():
    data = json.load(open(RAW, "r", encoding="utf-8"))

    key_types = [
        "task_round",
        "task_round_new",
        "task_transition",
        "task_round_transition",
        "round_transition",
        "qtype_round_transition",
        "task_qtype_round_transition",
    ]

    min_supports = [1, 2, 3, 5, 8, 10]
    min_nets = [1, 2, 3, 5]
    min_precisions = [0.50, 0.55, 0.60, 0.65, 0.70]

    rows = []
    best = None

    for kt in key_types:
        for ms in min_supports:
            for mn in min_nets:
                for mp in min_precisions:
                    res = run_policy(data, kt, ms, mn, mp)
                    row = {k: v for k, v in res.items() if k not in ("output", "stats", "good_keys")}
                    rows.append(row)
                    if best is None or res["acc"] > best["acc"]:
                        best = res

    rows = sorted(rows, key=lambda x: x["acc"], reverse=True)

    leaderboard = os.path.join(OUT_DIR, "group_selector_leaderboard.csv")
    with open(leaderboard, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "key_type", "min_support", "min_net", "min_precision",
            "samples", "correct", "acc", "changed",
            "r1_wrong_to_final_right", "r1_right_to_final_wrong",
            "net_gain", "num_good_groups", "selected_rounds"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    best_dir = os.path.join(OUT_DIR, "best_group_selector_output")
    os.makedirs(best_dir, exist_ok=True)
    json.dump(best["output"], open(os.path.join(best_dir, "output.json"), "w", encoding="utf-8"), ensure_ascii=False)

    groups_path = os.path.join(OUT_DIR, "best_good_groups.csv")
    with open(groups_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "n", "gain", "fix", "hurt"])
        for k in best["good_keys"]:
            st = best["stats"][k]
            w.writerow([k, st["n"], st["gain"], st["fix"], st["hurt"]])

    print("Top group selectors:")
    for r in rows[:15]:
        print(r)

    print("\nBEST:")
    print({k: v for k, v in best.items() if k not in ("output", "stats", "good_keys")})
    print("saved leaderboard:", leaderboard)
    print("saved best output:", os.path.join(best_dir, "output.json"))
    print("saved best groups:", groups_path)

if __name__ == "__main__":
    main()
