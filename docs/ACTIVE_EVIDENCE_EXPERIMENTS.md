# Running Active-Evidence Reasoning Experiments

This document describes the code changes needed to run the experiments for the paper written under the new paradigm of **Evidence-Sufficiency-Guided Active Video Reasoning**.

## What changed

The modified code keeps the inherited VideoMind backbone for:
- Planner,
- Grounder,
- Verifier,
- Answerer.

On top of this inherited backbone, it adds an inference-time **active evidence reasoning** layer with four components:
1. answer-conditioned evidence sufficiency estimation,
2. faithfulness-based answer acceptance,
3. evidence-acquisition actions, and
4. bounded multi-round selection.

## Modified / Added Files

### Modified
- `videomind/constants.py`
  - adds `REVERSE_QUESTION_PROMPT`
- `videomind/eval/infer_auto.py`
  - adds the active-evidence reasoning loop and ablation controls

### Added
- `videomind/utils/active_evidence.py`
  - token-level F1 overlap, answer text resolution, span expansion, and active-variant presets
- `videomind/eval/summarize_active.py`
  - summarizes executed-round statistics for round-distribution / efficiency analysis
- `scripts/evaluation/eval_active_2b.sh`
- `scripts/evaluation/eval_active_7b.sh`
- `scripts/evaluation/eval_active_ablation_2b.sh`
- `scripts/evaluation/eval_active_ablation_7b.sh`

### Backward-compatible aliases
- `videomind/utils/feedback.py`
- `videomind/eval/summarize_feedback.py`
- `scripts/evaluation/eval_feedback_*.sh`

These alias files are kept only for compatibility with older naming and can be ignored for the new paper.

## Core idea in code

After the inherited Verifier selects the top-1 segment, the new code performs:
1. answer generation from the current evidence state,
2. reverse question reconstruction from the generated answer,
3. token-level F1 overlap computation between the reconstructed question and the original question,
4. evidence sufficiency check using the faithfulness threshold,
5. evidence-acquisition action by symmetric temporal expansion when the current evidence is insufficient,
6. bounded multi-round reasoning and best-faithfulness round selection.

## Supported active variants

The main argument is:

```bash
--active_variant {baseline,es_only,es_fa,es_fa_ea,full}
```

The variants correspond to the paper ablations:

- `baseline`
  - inherited one-pass reasoning only
- `es_only`
  - answer-conditioned evidence sufficiency estimation only
- `es_fa`
  - evidence sufficiency estimation + faithfulness-based answer acceptance
- `es_fa_ea`
  - evidence sufficiency estimation + faithfulness-based acceptance + evidence-acquisition action, final answer from the last executed round
- `full`
  - full active-evidence reasoning framework with best-faithfulness round selection

Backward-compatible aliases are also supported:
- `qr_only -> es_only`
- `qr_oa -> es_fa`
- `qr_oa_br -> es_fa_ea`

## Default hyperparameters

The modified code uses the following defaults:
- sufficiency threshold: `0.8`
- boundary expansion ratio: `0.1`
- maximum rounds: `3`
- sufficiency score: token-level F1 overlap after lowercasing and punctuation removal
- selection rule for `full`: `best_faithfulness`

These values can be overridden through:

```bash
--sufficiency_threshold 0.8
--boundary_expand_ratio 0.1
--max_rounds 3
--selection_rule best_faithfulness
```


## Running on a specific GPU

If GPU 0 is occupied and you want to run on GPU 1 only, you do not need to change the Python code.
The updated shell scripts now accept an optional GPU argument as the last positional argument.

Examples:

```bash
# Run the full 2B model on GPU 1 only
bash scripts/evaluation/eval_active_2b.sh nextqa valid full 1

# Run all 2B ablations on GPU 1 only
bash scripts/evaluation/eval_active_ablation_2b.sh nextqa valid 1

# Run on multiple GPUs explicitly
bash scripts/evaluation/eval_active_2b.sh nextqa valid full 1,2
```

If you prefer, you can still use environment variables:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluation/eval_active_2b.sh nextqa valid full
```

The updated scripts now use a safer default of a single GPU when no GPU list is provided explicitly.

## Main commands

### Full model (2B)

```bash
bash scripts/evaluation/eval_active_2b.sh <dataset> [split] [variant] [gpu_devices]
```

Example:

```bash
bash scripts/evaluation/eval_active_2b.sh nextqa valid full
```

### Full model (7B)

```bash
bash scripts/evaluation/eval_active_7b.sh <dataset> [split] [variant] [gpu_devices]
```

### Run all ablations (2B)

```bash
bash scripts/evaluation/eval_active_ablation_2b.sh <dataset> [split] [gpu_devices]
```

### Run all ablations (7B)

```bash
bash scripts/evaluation/eval_active_ablation_7b.sh <dataset> [split] [gpu_devices]
```

## Output files

Each run saves prediction files in:
- `outputs_2b_active/<dataset>_<split>_<variant>/`
- `outputs_7b_active/<dataset>_<split>_<variant>/`

The output JSON now contains additional fields such as:
- `active_config`
- `evidence_rounds`
- `executed_rounds`
- `accepted_round`
- `selected_round`
- `selected_evidence_state`
- `selected_span`
- `final_answer_text`
- `reverse_question`
- `faithfulness_score`

These fields are intended for:
- main comparisons,
- framework ablations,
- sufficiency-threshold / boundary-ratio / max-round analysis,
- round-distribution statistics,
- qualitative case studies.

## Active summary for Fig. 3 / efficiency analysis

After each active-evidence run, the wrapper scripts automatically execute:

```bash
python videomind/eval/summarize_active.py <pred_path>
```

This produces `active_summary.json` with:
- total samples,
- average executed rounds,
- average selected round,
- average faithfulness score,
- executed-round histogram,
- selected-round histogram,
- accepted-round histogram.

These statistics can directly support:
- the round-distribution figure,
- the efficiency table,
- the stopping-round discussion.

## Recommended experiment order

To match the new paper structure, the recommended order is:

1. main comparison:
   - `baseline`
   - `full`
2. framework validation:
   - `baseline`, `es_only`, `es_fa`, `es_fa_ea`, `full`
3. sufficiency-threshold analysis:
   - vary `--sufficiency_threshold`
4. evidence-acquisition analysis:
   - vary `--boundary_expand_ratio`
5. bounded-policy analysis:
   - vary `--max_rounds`
   - compare `--selection_rule last`, `first_sufficient`, `best_faithfulness`
6. qualitative analysis:
   - inspect `evidence_rounds` and `selected_evidence_state`

## Practical limitation

The current implementation is a local instantiation of the broader active-evidence framework. It assumes that the initial top-1 segment roughly captures the semantic center of the relevant event and improves answer faithfulness by expanding temporal context around that center. Cases where the truly relevant evidence lies in a completely disjoint temporal region are not fully addressed by this implementation.
