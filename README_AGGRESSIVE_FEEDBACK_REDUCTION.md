# Aggressive Feedback Reduction Update

This version is tuned for the case where the baseline QA accuracy is already high and feedback previously produced many unnecessary second-round executions.

Main changes in `videomind/eval/infer_auto.py`:

- Default `candidate_pool_size`: 1
- Default `sufficiency_threshold`: 0.48
- Default `sufficiency_beta`: 0.25
- Default `acceptance_margin`: 0.12
- Default `verifier_accept_threshold`: 0.48
- Default `verifier_skip_reverse_threshold`: 0.46
- Default `min_feedback_gap`: 0.16
- Default `feedback_verifier_floor`: 0.42
- Default `planner_max_pixels`: `24 * 28 * 28`

Policy change:

- Accept directly if any usable confidence signal is above `verifier_accept_threshold`.
- Trigger feedback only when the effective confidence is clearly lower than `feedback_verifier_floor`.
- Borderline samples are stopped at round 1 to avoid changing already-correct answers.

Recommended command is included in the chat response.
