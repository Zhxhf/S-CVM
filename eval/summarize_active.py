# Copyright (c) 2026 OpenAI adaptation. Licensed under the BSD-3-Clause License.

import argparse
import json
from collections import Counter

import nncore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    parser.add_argument('--out_name', default='active_summary.json')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True)
    samples = []
    for path in paths:
        samples.extend(nncore.load(path))

    total = len(samples)
    executed = [int(s.get('executed_rounds', 1) or 1) for s in samples]
    selected = [int(s.get('selected_round', s.get('executed_rounds', 1)) or 1) for s in samples]
    accepted = [s.get('accepted_round') for s in samples]
    faithfulness = [s.get('faithfulness_score') for s in samples if s.get('faithfulness_score') is not None]
    variants = Counter([s.get('active_config', s.get('feedback_config', {})).get('variant', 'unknown') for s in samples])

    hist_executed = Counter(executed)
    hist_selected = Counter(selected)
    hist_accepted = Counter([a for a in accepted if a is not None])

    summary = dict(
        total_samples=total,
        variants=dict(variants),
        average_executed_rounds=sum(executed) / max(total, 1),
        average_selected_round=sum(selected) / max(total, 1),
        average_faithfulness_score=sum(faithfulness) / max(len(faithfulness), 1) if faithfulness else None,
        executed_round_histogram={str(k): int(v) for k, v in sorted(hist_executed.items())},
        selected_round_histogram={str(k): int(v) for k, v in sorted(hist_selected.items())},
        accepted_round_histogram={str(k): int(v) for k, v in sorted(hist_accepted.items())},
    )

    out_path = nncore.join(args.pred_path, args.out_name)
    nncore.dump(summary, out_path)
    print(json.dumps(summary, indent=2))
    print(f'\nSaved summary to: {out_path}')


if __name__ == '__main__':
    main()
