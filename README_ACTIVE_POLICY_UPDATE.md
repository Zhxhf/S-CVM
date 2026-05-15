# Active Evidence Policy Update

This package contains a code-level upgrade of the inference-time active-evidence policy for VideoMind.

## Main changes

- top-K candidate pool initialization instead of hard top-1 only
- sufficiency score = faithfulness score + verifier confidence
- feedback-driven action selection
- candidate switching when current evidence is likely mismatched
- optional two-segment joint reasoning for causal / temporal / dispersed evidence cases
- fallback from two-segment reasoning to union-span reasoning if the backend cannot process two video clips in one prompt

## Key modified files

- `videomind/eval/infer_auto.py`
- `videomind/utils/active_evidence.py`
- `videomind/constants.py`

## New important arguments

- `--candidate_pool_size 3`
- `--score_alpha 0.5`
- `--sufficiency_beta 0.7`
- `--pair_diversity_weight 0.35`
- `--disable_pair_fallback_union`

## Notes

The `full` variant now enables:

- candidate pool
- candidate switching
- feedback policy
- two-segment merge
- bounded active reasoning with best-faithfulness selection

If your runtime backend does not stably support two video clips in a single prompt, keep the default fallback enabled so the code will automatically fall back to the union span of the two segments.
