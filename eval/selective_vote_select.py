import argparse
import copy
import json
import os
import re
from collections import Counter

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

def uniq_key(r):
    cid = r.get('candidate_id')
    span = r.get('span')
    if cid is not None:
        return str(cid)
    if span is not None:
        return str(span)
    return str(r.get('round'))

def choose_round(sample, mode='selective_dedup'):
    rounds = sample.get('evidence_rounds') or []
    options = sample.get('options')
    if not rounds:
        return None

    r1 = rounds[0]
    r1_pred = pred_letter(r1.get('response'), r1.get('answer_text'), options)

    # 去重：R5 日志中 round3/4/5 经常是同一个 single_2，不能重复算三票
    unique_rounds = []
    seen = set()
    for r in rounds:
        k = uniq_key(r)
        if k in seen:
            continue
        seen.add(k)
        unique_rounds.append(r)

    preds = []
    pred_to_round = {}
    for r in unique_rounds:
        p = pred_letter(r.get('response'), r.get('answer_text'), options)
        if p:
            preds.append(p)
            pred_to_round.setdefault(p, r)

    if not preds:
        return r1

    cnt = Counter(preds)
    best_pred, best_count = cnt.most_common(1)[0]
    first_count = cnt.get(r1_pred, 0)

    r1_ver = r1.get('verifier_score')
    r1_suff = r1.get('sufficiency_score')
    r1_faith = r1.get('faithfulness_score')

    r1_ver = 0.5 if r1_ver is None else float(r1_ver)
    r1_suff = 0.5 if r1_suff is None else float(r1_suff)
    r1_faith = 0.5 if r1_faith is None else float(r1_faith)

    # 第一轮是否“足够稳”：稳则尽量不改，避免误伤
    r1_high_conf = (
        r1_ver >= 0.95 and
        first_count >= best_count
    )

    # 保守选择策略：
    # 1. 平票保留第一轮；
    # 2. 第一轮高置信保留第一轮；
    # 3. 只有去重后的候选投票明确压过第一轮，才替换；
    # 4. 替换时选择该选项第一次出现的轮次。
    if mode == 'selective_dedup':
        if best_pred != r1_pred and best_count > first_count and not r1_high_conf:
            return pred_to_round[best_pred]
        return r1

    # 更激进一点：只要去重投票胜出就替换
    if mode == 'dedup_vote':
        if best_pred != r1_pred and best_count > first_count:
            return pred_to_round[best_pred]
        return r1

    # 更保守：必须至少 2 个不同候选支持同一个非第一轮答案
    if mode == 'strong_dedup':
        if best_pred != r1_pred and best_count >= 2 and best_count > first_count:
            return pred_to_round[best_pred]
        return r1

    return r1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--mode', default='selective_dedup',
                    choices=['selective_dedup', 'dedup_vote', 'strong_dedup'])
    args = ap.parse_args()

    src = os.path.join(args.input_dir, 'output.json')
    dst_dir = args.output_dir
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'output.json')

    data = json.load(open(src, 'r', encoding='utf-8'))
    new_data = []

    changed = 0
    selected_counter = Counter()

    for s in data:
        t = copy.deepcopy(s)
        chosen = choose_round(t, args.mode)

        if chosen is not None:
            old_round = t.get('selected_round', 1)
            new_round = chosen.get('round', 1)

            if new_round != old_round:
                changed += 1

            t['selected_round'] = new_round
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
            t['selective_vote_mode'] = args.mode

            selected_counter[str(new_round)] += 1

        new_data.append(t)

    json.dump(new_data, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
    print('saved:', dst)
    print('samples:', len(new_data))
    print('changed selected round:', changed)
    print('selected round histogram:', dict(selected_counter))

if __name__ == '__main__':
    main()
