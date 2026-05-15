# Copyright (c) 2026 OpenAI adaptation. Licensed under the BSD-3-Clause License.

import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple


_ALIAS_TO_VARIANT = {
    'baseline': 'baseline',
    'es_only': 'es_only',
    'es_fa': 'es_fa',
    'es_fa_ea': 'es_fa_ea',
    'full': 'full',
    # backward-compatible aliases from the older feedback-loop naming
    'qr_only': 'es_only',
    'qr_oa': 'es_fa',
    'qr_oa_br': 'es_fa_ea',
}

_SELECTION_ALIAS = {
    'last': 'last',
    'first_sufficient': 'first_sufficient',
    'first_accepted': 'first_sufficient',
    'best_faithfulness': 'best_faithfulness',
    'best_overlap': 'best_faithfulness',
}

_FEEDBACK_LABELS = (
    'subject', 'action', 'local_context', 'reason', 'result', 'temporal_order', 'dialogue', 'uncertain'
)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def tokenize_text(text: Optional[str]) -> List[str]:
    text = normalize_text(text)
    return text.split() if text else []



def token_f1_overlap(question: Optional[str], reconstructed_question: Optional[str]) -> float:
    q_tokens = tokenize_text(question)
    r_tokens = tokenize_text(reconstructed_question)
    if len(q_tokens) == 0 or len(r_tokens) == 0:
        return 0.0

    q_cnt: Dict[str, int] = {}
    for token in q_tokens:
        q_cnt[token] = q_cnt.get(token, 0) + 1

    overlap = 0
    for token in r_tokens:
        if q_cnt.get(token, 0) > 0:
            overlap += 1
            q_cnt[token] -= 1

    precision = overlap / len(r_tokens)
    recall = overlap / len(q_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



def extract_option_index(response: Optional[str], num_options: int) -> Optional[int]:
    if not response:
        return None
    text = str(response).strip()
    match = re.search(r'\b([A-Za-z])\b', text)
    if match is None:
        return None
    idx = ord(match.group(1).lower()) - ord('a')
    if 0 <= idx < num_options:
        return idx
    return None



def resolve_answer_text(response: Optional[str], options: Optional[Sequence[str]]) -> str:
    if response is None:
        return ''
    if options:
        idx = extract_option_index(response, len(options))
        if idx is not None:
            return str(options[idx]).strip()
    return str(response).strip()



def clamp_span(span: Sequence[float], duration: float) -> Tuple[float, float]:
    start, end = float(span[0]), float(span[1])
    start = max(0.0, min(duration, start))
    end = max(0.0, min(duration, end))
    return min(start, end), max(start, end)



def expand_span(span: Sequence[float], duration: float, ratio: float) -> Tuple[float, float]:
    start, end = clamp_span(span, duration)
    delta = end - start
    start = max(0.0, start - ratio * delta)
    end = min(duration, end + ratio * delta)
    return start, end



def expand_segments(segments: Sequence[Sequence[float]], duration: float, ratio: float) -> List[List[float]]:
    return [list(expand_span(seg, duration, ratio)) for seg in segments]



def span_iou(a: Sequence[float], b: Sequence[float]) -> float:
    a0, a1 = clamp_span(a, max(float(a[1]), float(b[1]), 1.0))
    b0, b1 = clamp_span(b, max(float(a[1]), float(b[1]), 1.0))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union



def segment_diversity(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - span_iou(a, b)



def union_span(segments: Sequence[Sequence[float]], duration: float) -> List[float]:
    if not segments:
        return [0.0, duration]
    starts, ends = zip(*[clamp_span(seg, duration) for seg in segments])
    return [max(0.0, min(starts)), min(duration, max(ends))]





def infer_temporal_preference(question: Optional[str],
                              reconstructed_question: Optional[str] = None,
                              feedback_label: Optional[str] = None) -> str:
    text = normalize_text((question or '') + ' ' + (reconstructed_question or ''))
    label = normalize_text(feedback_label)
    if label == 'reason':
        return 'left'
    if label == 'result':
        return 'right'
    left_markers = [' before ', ' previous ', ' earlier ', ' prior ', ' leading up', ' cause ', ' why ']
    right_markers = [' after ', ' next ', ' then ', ' later ', ' result ', ' outcome ']
    has_left = any(marker.strip() in text for marker in left_markers)
    has_right = any(marker.strip() in text for marker in right_markers)
    if has_left and not has_right:
        return 'left'
    if has_right and not has_left:
        return 'right'
    return 'both'


def expand_span_directional(span: Sequence[float], duration: float, ratio: float, direction: str = 'both') -> Tuple[float, float]:
    start, end = clamp_span(span, duration)
    delta = max(1e-6, end - start)
    direction = normalize_text(direction) or 'both'
    if direction in {'left', 'expandleft'}:
        start = max(0.0, start - ratio * delta)
    elif direction in {'right', 'expandright'}:
        end = min(duration, end + ratio * delta)
    else:
        start = max(0.0, start - ratio * delta)
        end = min(duration, end + ratio * delta)
    return start, end


def expand_segments_directional(segments: Sequence[Sequence[float]], duration: float, ratio: float,
                               direction: str = 'both') -> List[List[float]]:
    if not segments:
        return []
    if len(segments) == 1:
        return [list(expand_span_directional(segments[0], duration, ratio, direction))]
    # For multi-segment evidence, apply directional growth to the earliest/latest segment and mild growth to the rest.
    ordered = [list(clamp_span(seg, duration)) for seg in segments]
    if direction == 'left':
        ordered[0] = list(expand_span_directional(ordered[0], duration, ratio, 'left'))
    elif direction == 'right':
        ordered[-1] = list(expand_span_directional(ordered[-1], duration, ratio, 'right'))
    else:
        ordered[0] = list(expand_span_directional(ordered[0], duration, ratio, 'left'))
        ordered[-1] = list(expand_span_directional(ordered[-1], duration, ratio, 'right'))
    return ordered

def round_span(span: Sequence[float], unit: float) -> List[float]:
    if unit <= 0:
        return [float(span[0]), float(span[1])]
    return [round(float(span[0]) / unit) * unit, round(float(span[1]) / unit) * unit]



def round_segments(segments: Sequence[Sequence[float]], unit: float) -> List[List[float]]:
    return [round_span(seg, unit) for seg in segments]



def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default



def minmax_normalize(values: Sequence[Any], default: float = 0.5) -> List[float]:
    vals = [safe_float(v, 0.0) for v in values]
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-8:
        return [default for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]



def combine_candidate_score(grounder_score: float,
                            verifier_score: float,
                            alpha: float = 0.5) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * safe_float(grounder_score) + (1.0 - alpha) * safe_float(verifier_score)



def combine_sufficiency_score(faithfulness_score: Optional[float],
                              verifier_score: Optional[float],
                              beta: float = 0.7) -> Optional[float]:
    if faithfulness_score is None and verifier_score is None:
        return None
    if faithfulness_score is None:
        return safe_float(verifier_score)
    if verifier_score is None:
        return safe_float(faithfulness_score)
    beta = min(max(float(beta), 0.0), 1.0)
    return beta * safe_float(faithfulness_score) + (1.0 - beta) * safe_float(verifier_score)



def parse_feedback_label(text: Optional[str]) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return 'uncertain'
    alias_patterns = {
        'local_context': ['local context', 'context', 'surrounding', 'before after', 'neighborhood'],
        'temporal_order': ['temporal order', 'order', 'before', 'after', 'sequence'],
        'dialogue': ['dialogue', 'subtitle', 'speech', 'conversation'],
        'subject': ['subject', 'person', 'object identity', 'identity'],
        'action': ['action', 'motion', 'behavior', 'doing'],
        'reason': ['reason', 'why', 'cause', 'because'],
        'result': ['result', 'outcome', 'effect', 'afterward'],
    }
    for label in _FEEDBACK_LABELS:
        if label in normalized:
            return label
    for label, patterns in alias_patterns.items():
        for pat in patterns:
            if pat in normalized:
                return label
    return 'uncertain'



def heuristic_feedback_label(question: Optional[str],
                             reconstructed_question: Optional[str] = None) -> str:
    q = normalize_text(question)
    rq = normalize_text(reconstructed_question)
    text = q + ' ' + rq
    if any(k in text for k in ['why', 'cause', 'because', 'reason']):
        return 'reason'
    if any(k in text for k in ['what happened after', 'after', 'before', 'then', 'next', 'first', 'finally']):
        return 'temporal_order'
    if any(k in text for k in ['say', 'said', 'tell', 'conversation', 'subtitle', 'speak']):
        return 'dialogue'
    if any(k in text for k in ['who', 'which person', 'which object']):
        return 'subject'
    if any(k in text for k in ['doing', 'do ', 'did ', 'what is', 'what was']):
        return 'action'
    return 'local_context'



def resolve_selection_rule(name: Optional[str]) -> str:
    if name is None:
        return 'last'
    if name not in _SELECTION_ALIAS:
        raise ValueError(f'unsupported selection rule: {name}')
    return _SELECTION_ALIAS[name]



def resolve_active_variant(name: str,
                           max_rounds: int,
                           selection_rule: Optional[str] = None) -> Dict[str, Any]:
    if name not in _ALIAS_TO_VARIANT:
        raise ValueError(f'unsupported active-evidence variant: {name}')

    canonical = _ALIAS_TO_VARIANT[name]
    cfg = dict(
        variant=canonical,
        use_sufficiency_estimation=False,
        use_faithfulness_acceptance=False,
        use_evidence_acquisition=False,
        use_candidate_pool=False,
        use_candidate_switch=False,
        use_merge_two_segments=False,
        use_feedback_policy=False,
        selection_rule='last',
        max_rounds=1,
    )

    if canonical == 'es_only':
        cfg['use_sufficiency_estimation'] = True
    elif canonical == 'es_fa':
        cfg['use_sufficiency_estimation'] = True
        cfg['use_faithfulness_acceptance'] = True
    elif canonical == 'es_fa_ea':
        cfg['use_sufficiency_estimation'] = True
        cfg['use_faithfulness_acceptance'] = True
        cfg['use_evidence_acquisition'] = True
        cfg['max_rounds'] = max(1, int(max_rounds))
    elif canonical == 'full':
        cfg['use_sufficiency_estimation'] = True
        cfg['use_faithfulness_acceptance'] = True
        cfg['use_evidence_acquisition'] = True
        cfg['use_candidate_pool'] = True
        cfg['use_candidate_switch'] = True
        cfg['use_merge_two_segments'] = True
        cfg['use_feedback_policy'] = True
        cfg['selection_rule'] = 'best_faithfulness'
        cfg['max_rounds'] = max(1, int(max_rounds))

    if selection_rule is not None:
        cfg['selection_rule'] = resolve_selection_rule(selection_rule)
    return cfg
