from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..types import JudgeResult, NormalizedExample
from .parsing import extract_json_object


def _normalize_choice_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^[^A-Za-z0-9_]*", "", value)
    value = re.sub(r"[^A-Za-z0-9_].*$", "", value)
    return value.upper()


def _parse_answer_and_reasoning(raw_text: str) -> Tuple[Optional[str], str, bool, Dict[str, Any]]:
    parse_error = False
    try:
        obj = extract_json_object(raw_text)
    except Exception:
        parse_error = True
        return None, "Failed to parse JSON candidate answer.", parse_error, {}

    raw_answer = obj.get("answer")
    reasoning = str(obj.get("reasoning", "")).strip()

    answer_value: Optional[str] = None
    if isinstance(raw_answer, str) and raw_answer.strip():
        answer_value = raw_answer.strip()
    elif isinstance(raw_answer, list):
        first = next((x for x in raw_answer if isinstance(x, str) and x.strip()), None)
        if first:
            answer_value = first.strip()

    return answer_value, reasoning, parse_error, obj


def _expected_choice_ids(example: NormalizedExample) -> List[str]:
    vals = example.metadata.get("correct_choice_ids")
    if isinstance(vals, list):
        out = [str(x).strip() for x in vals if isinstance(x, str) and x.strip()]
        if out:
            return out
    raise ValueError(
        f"MCQ example '{example.id}' is missing `metadata.correct_choice_ids`; "
        "rebuild canonical dataset inputs before running."
    )


def grade_mcq_output(
    example: NormalizedExample, candidate_text: str, pass_threshold: float
) -> JudgeResult:
    expected_ids = _expected_choice_ids(example)
    expected_norm = {_normalize_choice_id(x): x for x in expected_ids}

    answer_raw, reasoning, parse_error, parsed_obj = _parse_answer_and_reasoning(candidate_text)
    selected_id: Optional[str] = None
    if answer_raw:
        selected_norm = _normalize_choice_id(answer_raw)
        selected_id = expected_norm.get(selected_norm, answer_raw.strip())

    exact_match = 1.0 if selected_id in expected_ids and bool(expected_ids) else 0.0
    passed = exact_match >= pass_threshold

    rationale_parts = []
    if reasoning:
        rationale_parts.append(reasoning)
    rationale_parts.append(f"Selected={selected_id or '(none)'}; expected={expected_ids or '(none)'}")
    if parse_error:
        rationale_parts.append("Parse error: candidate output was not valid JSON.")

    return JudgeResult(
        score=exact_match,
        passed=passed,
        rationale=" | ".join(rationale_parts),
        criteria={"exact_match": exact_match},
        raw={
            "parsed_candidate": parsed_obj,
            "selected_answer": selected_id,
            "expected_choice_ids": expected_ids,
            "parse_error": parse_error,
        },
        parse_error=parse_error,
    )
