# Low-feedback version: target feedback rate < 20%

This version changes `videomind/eval/infer_auto.py` to aggressively reduce unnecessary feedback rounds.

Main changes:
- Default `feedback_budget_ratio=0.18`, a deterministic hard budget that prevents more than about 18% of samples from being eligible for feedback.
- Stricter `should_trigger_feedback`: feedback requires both budget allowance and clearly low confidence.
- More permissive first-round acceptance defaults:
  - `sufficiency_threshold=0.40`
  - `acceptance_margin=0.18`
  - `verifier_accept_threshold=0.40`
  - `verifier_skip_reverse_threshold=0.38`
  - `feedback_verifier_floor=0.30`
  - `min_feedback_gap=0.22`
- Reduced visual budget defaults for stability:
  - `planner_max_pixels=20*28*28`
  - `planner_max_frames=20`

Recommended run command is included in the ChatGPT response.
