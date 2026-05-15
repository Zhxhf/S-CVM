import json, os, re, csv, copy
from collections import Counter, defaultdict

RAW = "outputs_2b_active/nextqa_valid_full_sweep_r5_raw/output.json"
OUT_DIR = "outputs_2b_active/r5_selector_search"
os.makedirs(OUT_DIR, exist_ok=True)

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
        for i,opt in enumerate(options):
            opt = str(opt).strip()
            if ans_text.strip().lower() == opt.lower():
                return "ABCDE"[i]
            if opt.lower() in resp.lower():
                return "ABCDE"[i]
    return None

def uniq_rounds(rounds):
    out = []
    seen = set()
    for r in rounds:
        k = r.get('candidate_id') or str(r.get('span')) or str(r.get('round'))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def score_round(r):
    suff = r.get('sufficiency_score')
    faith = r.get('faithfulness_score')
    ver = r.get('verifier_score')
    score = 0.0
    w = 0.0
    if suff is not None:
        score += float(suff) * 0.5
        w += 0.5
    if faith is not None:
        score += float(faith) * 0.3
        w += 0.3
    if ver is not None:
        score += float(ver) * 0.2
        w += 0.2
    return score / w if w else 0.0

def choose_by_policy(sample, policy):
    rounds = sample.get('evidence_rounds') or []
    options = sample.get('options')
    if not rounds:
        return None

    r1 = rounds[0]
    r1_pred = pred_letter(r1.get('response'), r1.get('answer_text'), options)
    urs = uniq_rounds(rounds)

    if policy == "first":
        return r1

    if policy == "raw_vote":
        preds = []
        pred_to_round = {}
        for r in rounds:
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p:
                preds.append(p)
                pred_to_round.setdefault(p, r)
        if not preds:
            return r1
        cnt = Counter(preds)
        best, bestn = cnt.most_common(1)[0]
        if best != r1_pred and bestn > cnt.get(r1_pred, 0):
            return pred_to_round[best]
        return r1

    if policy == "dedup_vote":
        preds = []
        pred_to_round = {}
        for r in urs:
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p:
                preds.append(p)
                pred_to_round.setdefault(p, r)
        if not preds:
            return r1
        cnt = Counter(preds)
        best, bestn = cnt.most_common(1)[0]
        if best != r1_pred and bestn > cnt.get(r1_pred, 0):
            return pred_to_round[best]
        return r1

    if policy == "round2_if_diff":
        if len(urs) >= 2:
            r = urs[1]
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p and p != r1_pred:
                return r
        return r1

    if policy == "round3_if_diff":
        if len(urs) >= 3:
            r = urs[2]
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p and p != r1_pred:
                return r
        return r1

    if policy == "first_changed":
        for r in urs[1:]:
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p and p != r1_pred:
                return r
        return r1

    if policy == "min_score_diff":
        cands = []
        for r in urs[1:]:
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p and p != r1_pred:
                cands.append((score_round(r), r))
        if cands:
            return min(cands, key=lambda x: x[0])[1]
        return r1

    if policy == "max_verifier_diff":
        cands = []
        for r in urs[1:]:
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p and p != r1_pred:
                ver = r.get('verifier_score')
                ver = -1 if ver is None else float(ver)
                cands.append((ver, r))
        if cands:
            return max(cands, key=lambda x: x[0])[1]
        return r1

    return r1

def apply_choice(sample, chosen):
    t = copy.deepcopy(sample)
    if chosen is None:
        return t
    t['selected_round'] = chosen.get('round', 1)
    t['accepted_round'] = chosen.get('round', t.get('accepted_round'))
    t['response'] = chosen.get('response', t.get('response'))
    t['answerer_response'] = chosen.get('response', t.get('answerer_response'))
    t['final_answer_text'] = chosen.get('answer_text', t.get('final_answer_text'))
    t['reverse_question'] = chosen.get('reverse_question')
    t['faithfulness_score'] = chosen.get('faithfulness_score')
    t['sufficiency_score'] = chosen.get('sufficiency_score')
    t['selected_span'] = chosen.get('span', t.get('selected_span'))
    t['selected_evidence_state'] = chosen.get('evidence_state', t.get('selected_evidence_state'))
    t['selected_candidate_id'] = chosen.get('candidate_id', t.get('selected_candidate_id'))
    t['selected_candidate_kind'] = chosen.get('candidate_kind', t.get('selected_candidate_kind'))
    return t

def eval_policy(data, policy, gate_mode="none"):
    total = len(data)
    correct = 0
    changed = 0
    r1_wrong_to_right = 0
    r1_right_to_wrong = 0
    selected_rounds = Counter()

    # gate_mode:
    # none: 所有样本都按 policy
    # task_positive: 只在该 policy 对某个 task 净收益为正时才替换
    # task_round_positive: 只在该 policy 对某个 task+round 净收益为正时才替换

    chosen_cache = []
    task_net = Counter()
    task_round_net = Counter()

    for s in data:
        rounds = s.get('evidence_rounds') or []
        options = s.get('options')
        gt = s.get('ans')
        task = s.get('task')
        r1 = rounds[0] if rounds else None
        r1_pred = pred_letter(r1.get('response'), r1.get('answer_text'), options) if r1 else pred_letter(s.get('response'), s.get('final_answer_text'), options)

        chosen = choose_by_policy(s, policy)
        cpred = pred_letter(chosen.get('response'), chosen.get('answer_text'), options) if chosen else r1_pred
        cround = chosen.get('round', 1) if chosen else 1

        old_correct = (r1_pred == gt)
        new_correct = (cpred == gt)
        delta = int(new_correct) - int(old_correct)

        if cpred != r1_pred:
            task_net[task] += delta
            task_round_net[(task, cround)] += delta

        chosen_cache.append((s, chosen, r1_pred, cpred, cround, old_correct, new_correct))

    pos_tasks = {t for t,v in task_net.items() if v > 0}
    pos_task_rounds = {tr for tr,v in task_round_net.items() if v > 0}

    output = []
    for s, chosen, r1_pred, cpred, cround, old_correct, new_correct in chosen_cache:
        task = s.get('task')
        use = True

        if gate_mode == "task_positive":
            use = task in pos_tasks
        elif gate_mode == "task_round_positive":
            use = (task, cround) in pos_task_rounds

        if not use:
            chosen = (s.get('evidence_rounds') or [None])[0]
            cpred = r1_pred
            cround = 1
            new_correct = old_correct

        if cpred == s.get('ans'):
            correct += 1

        if cpred != r1_pred:
            changed += 1
            if not old_correct and new_correct:
                r1_wrong_to_right += 1
            if old_correct and not new_correct:
                r1_right_to_wrong += 1

        selected_rounds[str(cround)] += 1
        output.append(apply_choice(s, chosen))

    return {
        "policy": policy,
        "gate_mode": gate_mode,
        "samples": total,
        "correct": correct,
        "acc": correct / total * 100,
        "changed": changed,
        "r1_wrong_to_right": r1_wrong_to_right,
        "r1_right_to_wrong": r1_right_to_wrong,
        "net_gain": r1_wrong_to_right - r1_right_to_wrong,
        "selected_rounds": dict(selected_rounds),
        "output": output,
    }

def main():
    data = json.load(open(RAW, 'r', encoding='utf-8'))
    policies = [
        "first",
        "raw_vote",
        "dedup_vote",
        "round2_if_diff",
        "round3_if_diff",
        "first_changed",
        "min_score_diff",
        "max_verifier_diff",
    ]
    gate_modes = ["none", "task_positive", "task_round_positive"]

    rows = []
    best = None

    for p in policies:
        for g in gate_modes:
            if p == "first" and g != "none":
                continue
            res = eval_policy(data, p, g)
            row = {k:v for k,v in res.items() if k != "output"}
            rows.append(row)
            if best is None or res["acc"] > best["acc"]:
                best = res

    rows = sorted(rows, key=lambda x: x["acc"], reverse=True)

    csv_path = os.path.join(OUT_DIR, "selector_leaderboard.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["policy","gate_mode","samples","correct","acc","changed","r1_wrong_to_right","r1_right_to_wrong","net_gain","selected_rounds"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    best_dir = os.path.join(OUT_DIR, "best_selector_output")
    os.makedirs(best_dir, exist_ok=True)
    json.dump(best["output"], open(os.path.join(best_dir, "output.json"), "w", encoding="utf-8"), ensure_ascii=False)

    print("Top selectors:")
    for r in rows[:10]:
        print(r)
    print("\nBEST:", {k:v for k,v in best.items() if k != "output"})
    print("saved leaderboard:", csv_path)
    print("saved best output:", os.path.join(best_dir, "output.json"))

if __name__ == "__main__":
    main()
