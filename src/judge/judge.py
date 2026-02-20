from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, List

from src.types import JudgeResult, LLMMessage, NormalizedExample

from .parsing import extract_json_object
from .policies import (
    APEX_V1_GRADING_PROMPT_TEMPLATE,
    LEXAM_JUDGE_JSON_OUTPUT_INSTRUCTION,
    LEXAM_JUDGE_SYSTEM,
    LEXAM_JUDGE_USER_PROMPT,
    PRBENCH_GRADER_TEMPLATE,
    get_judge_policy_handler,
)
from .policies.shared import criterion_weight


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _criterion_aliases(item: Dict[str, Any], criterion_index: int) -> set[str]:
    criterion_id = str(item.get("id", f"criterion_{criterion_index}")).strip()
    criterion_title = str(item.get("title", f"Criterion {criterion_index}")).strip()
    return {
        _normalize_key(criterion_id),
        _normalize_key(criterion_title),
        _normalize_key(f"criterion_{criterion_index}"),
        _normalize_key(f"criterion {criterion_index}"),
    }


def resolve_rubric_criterion_score(
    criteria: Dict[str, float],
    criterion: Dict[str, Any],
    criterion_index: int,
    fallback_score: float = 0.0,
) -> tuple[float, bool]:
    if not isinstance(criteria, dict) or not criteria:
        return max(0.0, min(1.0, float(fallback_score))), False

    normalized_scores = {_normalize_key(k): float(v) for k, v in criteria.items()}
    for alias in _criterion_aliases(criterion, criterion_index):
        if alias and alias in normalized_scores:
            return max(0.0, min(1.0, float(normalized_scores[alias]))), True

    return max(0.0, min(1.0, float(fallback_score))), False


def _parse_binary_score(raw_value: Any) -> tuple[float, bool]:
    if isinstance(raw_value, bool):
        return (1.0 if raw_value else 0.0), True
    if isinstance(raw_value, (int, float)):
        return (1.0 if float(raw_value) >= 0.5 else 0.0), True
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "met"}:
            return 1.0, True
        if normalized in {"0", "false", "no", "not_met", "not met"}:
            return 0.0, True
    return 0.0, False


def build_judge_messages(
    example: NormalizedExample,
    model_output: str,
    pass_threshold: float,
) -> List[LLMMessage]:
    handler = get_judge_policy_handler(example.metadata.get("policy_id"))
    return handler.build_judge_messages(example, model_output, pass_threshold)


def build_rubric_criterion_judge_messages(
    example: NormalizedExample,
    model_output: str,
    criterion: Dict[str, Any],
    criterion_index: int,
    pass_threshold: float,
) -> List[LLMMessage]:
    handler = get_judge_policy_handler(example.metadata.get("policy_id"))
    return handler.build_rubric_criterion_judge_messages(
        example,
        model_output,
        criterion,
        criterion_index,
        pass_threshold,
    )


def parse_judge_output(raw_text: str, fallback_pass_threshold: float) -> JudgeResult:
    parse_error = False
    try:
        obj = extract_json_object(raw_text)
    except Exception:
        parse_error = True
        obj = {
            "score": 0.0,
            "pass": False,
            "rationale": "Failed to parse judge JSON output.",
            "criteria": {"overall": 0.0},
        }

    score_was_recognized = True
    if "score" in obj:
        score_raw = obj.get("score", 0.0)
    elif "result" in obj:
        score_raw, score_was_recognized = _parse_binary_score(obj.get("result"))
    elif "criteria_met" in obj:
        score_raw, score_was_recognized = _parse_binary_score(obj.get("criteria_met"))
    elif "grade" in obj:
        score_raw, score_was_recognized = _parse_binary_score(obj.get("grade"))
    else:
        score_raw = 0.0
        score_was_recognized = False

    if not score_was_recognized:
        parse_error = True

    score = float(score_raw)
    score = max(0.0, min(1.0, score))

    parsed_pass = obj.get("pass")
    if isinstance(parsed_pass, bool):
        passed = parsed_pass
    else:
        passed = score >= fallback_pass_threshold

    rationale = str(
        obj.get("rationale")
        or obj.get("reason")
        or obj.get("reasoning")
        or obj.get("explanation")
        or ""
    ).strip()

    raw_criteria = obj.get("criteria", {})
    criteria: Dict[str, float] = {}
    if isinstance(raw_criteria, dict):
        for key, value in raw_criteria.items():
            try:
                criteria[str(key)] = max(0.0, min(1.0, float(value)))
            except Exception:
                continue

    if not criteria:
        criteria = {"overall": score}

    return JudgeResult(
        score=score,
        passed=passed,
        rationale=rationale,
        criteria=criteria,
        raw=obj,
        parse_error=parse_error,
    )


def apply_weighted_rubric_score(
    parsed: JudgeResult,
    example: NormalizedExample,
    pass_threshold: float,
) -> JudgeResult:
    if example.judge_mode != "rubric":
        return parsed
    if not isinstance(example.rubric, list) or not example.rubric:
        return parsed
    if not isinstance(parsed.criteria, dict) or not parsed.criteria:
        return parsed

    weighted_items = []
    matched = 0
    for idx, item in enumerate(example.rubric, start=1):
        if not isinstance(item, dict):
            continue
        score_value, was_matched = resolve_rubric_criterion_score(
            criteria=parsed.criteria,
            criterion=item,
            criterion_index=idx,
            fallback_score=0.0,
        )
        if was_matched:
            matched += 1

        weight = criterion_weight(item)
        weighted_items.append((weight, score_value))

    if not weighted_items or matched == 0:
        return parsed

    raw_sum = sum(weight * score for weight, score in weighted_items)
    min_raw = sum(min(0.0, weight) for weight, _ in weighted_items)
    max_raw = sum(max(0.0, weight) for weight, _ in weighted_items)

    if max_raw <= min_raw:
        weighted_score = parsed.score
    else:
        weighted_score = (raw_sum - min_raw) / (max_raw - min_raw)

    normalized_points = float(weighted_score)
    clipped_points = max(0.0, min(1.0, normalized_points))
    weighted_score = clipped_points

    return replace(
        parsed,
        score=weighted_score,
        passed=weighted_score >= pass_threshold,
        raw={
            **(parsed.raw or {}),
            "deterministic_rubric_aggregation": {
                "applied": True,
                "matched_criteria": matched,
                "total_criteria": len(weighted_items),
                "raw_sum": raw_sum,
                "min_raw": min_raw,
                "max_raw": max_raw,
                "weighted_score": weighted_score,
                "normalized_points": normalized_points,
                "clipped_points": clipped_points,
            },
        },
    )


def apply_policy_score_postprocessing(
    parsed: JudgeResult,
    example: NormalizedExample,
    pass_threshold: float,
) -> JudgeResult:
    handler = get_judge_policy_handler(example.metadata.get("policy_id"))
    return handler.postprocess_judge_result(parsed, example, pass_threshold)
