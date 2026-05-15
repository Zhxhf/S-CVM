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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    src = os.path.join(args.input_dir, 'output.json')
    dst_dir = args.output_dir
    dst = os.path.join(dst_dir, 'output.json')
    os.makedirs(dst_dir, exist_ok=True)

    data = json.load(open(src, 'r', encoding='utf-8'))
    new_data = []

    changed = 0
    selected_round_counter = Counter()

    for sample in data:
        t = copy.deepcopy(sample)
        rounds = t.get('evidence_rounds') or []
        options = t.get('options')

        if not rounds:
            new_data.append(t)
            continue

        preds = []
        pred_to_first_round = {}

        for r in rounds:
            p = pred_letter(r.get('response'), r.get('answer_text'), options)
            if p:
                preds.append(p)
                pred_to_first_round.setdefault(p, r)

        if preds:
            first_pred = preds[0]
            cnt = Counter(preds)
            best_pred, best_count = cnt.most_common(1)[0]
            first_count = cnt.get(first_pred, 0)

            # 保守投票：只有其他选项票数严格超过第一轮答案，才替换。
            if best_pred != first_pred and best_count > first_count:
                chosen = pred_to_first_round[best_pred]
            else:
                chosen = rounds[0]

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
            t['vote_counts'] = dict(cnt)
            t['vote_first_pred'] = first_pred
            t['vote_selected_pred'] = pred_letter(t.get('response'), t.get('final_answer_text'), options)

            selected_round_counter[str(new_round)] += 1

        new_data.append(t)

    json.dump(new_data, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
    print('saved:', dst)
    print('samples:', len(new_data))
    print('changed selected round:', changed)
    print('selected_round_counter:', dict(selected_round_counter))

if __name__ == '__main__':
    main()
