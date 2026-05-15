# Feedback-rate tuned version (estimated target: 30%–40%)

This package was adjusted to reduce almost-always-triggered feedback in active-evidence reasoning while preserving the original answer quality as much as possible.

## Main changes

### Default inference parameters (`videomind/eval/infer_auto.py`)
- `--max_rounds 2`
- `--sufficiency_threshold 0.48`
- `--sufficiency_beta 0.25`
- `--candidate_pool_size 1`
- `--boundary_expand_ratio 0.08`

### New conservative gates
- `--acceptance_margin 0.12`
- `--verifier_accept_threshold 0.48`
- `--verifier_skip_reverse_threshold 0.46`
- `--min_feedback_gap 0.16`
- `--feedback_verifier_floor 0.42`
- `--sufficiency_eval_rounds 1`

## Behavior changes
1. High-verifier candidates can be accepted directly.
2. Reverse-question generation is skipped when verifier confidence is already high.
3. Sufficiency is evaluated only in the first round by default.
4. Feedback is triggered only when the candidate is clearly below the acceptance zone.
5. Expansion becomes slightly milder.

## Recommended command
```bash
CUDA_VISIBLE_DEVICES=1 python videomind/eval/infer_auto.py \
  --dataset nextqa \
  --split valid \
  --pred_path outputs_2b_active/nextqa_valid_full_tuned \
  --model_gnd_path model_zoo/VideoMind-2B \
  --model_ver_path model_zoo/VideoMind-2B \
  --model_pla_path model_zoo/VideoMind-2B \
  --model_ans_path model_zoo/VideoMind-2B \
  --active_variant full \
  --max_rounds 2 \
  --sufficiency_threshold 0.48 \
  --sufficiency_beta 0.25 \
  --candidate_pool_size 1 \
  --boundary_expand_ratio 0.08 \
  --acceptance_margin 0.12 \
  --verifier_accept_threshold 0.48 \
  --verifier_skip_reverse_threshold 0.46 \
  --min_feedback_gap 0.16 \
  --feedback_verifier_floor 0.42 \
  --sufficiency_eval_rounds 1 \
  --selection_rule best_faithfulness \
  --save_round_details
```

## Expected effect
This is an estimate, not a guarantee. The tuning is intentionally conservative and should typically move feedback from near-100% down toward a much lower range, often around the requested 30%–40% on datasets like NextQA when the original system was over-triggering.
