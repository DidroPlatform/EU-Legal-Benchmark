from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Dict, List

from ..data.policies import get_policy
from ..types import JudgeResult, LLMMessage, NormalizedExample
from .parsing import extract_json_object


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _criterion_weight(item: Dict[str, Any]) -> float:
    weight = item.get("weight")
    if isinstance(weight, (int, float)):
        return float(weight)

    annotations = item.get("annotations", {})
    if isinstance(annotations, dict):
        for key in (
            "critically_important_weight",
            "important_weight",
            "slightly_important_weight",
            "critically_detrimental_weight",
            "detrimental_weight",
            "slightly_detrimental_weight",
        ):
            value = annotations.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return 1.0


def _rubric_for_prompt(rubric: List[Dict[str, Any]] | None) -> str:
    if not rubric:
        return "No rubric provided."

    lines = []
    for i, item in enumerate(rubric, start=1):
        title = str(item.get("title", f"Criterion {i}"))
        criterion_id = str(item.get("id", f"criterion_{i}"))
        weight = _criterion_weight(item)
        lines.append(f"- {criterion_id}: {title} (weight_hint={weight})")
    return "\n".join(lines)


def _rubric_criterion_for_prompt(item: Dict[str, Any], criterion_index: int) -> str:
    title = str(item.get("title", f"Criterion {criterion_index}"))
    criterion_id = str(item.get("id", f"criterion_{criterion_index}"))
    weight = _criterion_weight(item)
    return f"- {criterion_id}: {title} (weight_hint={weight})"


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


def _judge_instruction(example: NormalizedExample, pass_threshold: float) -> str:
    if example.judge_mode == "mcq":
        return (
            "Evaluate as multiple-choice grading. Infer the option selected by the candidate answer. "
            "Give score=1.0 only if selected option matches the reference option exactly; else score=0.0. "
            "Set criteria as {'exact_match': score}. "
            f"Set pass=true only when score >= {pass_threshold:.3f}."
        )
    if example.judge_mode == "rubric":
        policy = get_policy(example.metadata.get("policy_id"))
        if policy.rubric_judge_style == "criterion_binary":
            return (
                "Evaluate each rubric criterion independently with a binary score (1 if met, 0 if not met). "
                "Set criteria as a mapping from criterion IDs to 0 or 1. "
                "Set overall score as weighted criterion fulfillment in [0,1]. "
                f"Set pass=true when score >= {pass_threshold:.3f}."
            )
        return (
            "Evaluate against the rubric criteria. Score should reflect weighted rubric fulfillment and overall quality. "
            "Populate criteria with criterion-level scores in [0,1]. "
            f"Set pass=true when score >= {pass_threshold:.3f}."
        )
    return (
        "Evaluate against reference answer and context for factual/semantic correctness. "
        "Populate criteria as {'overall': score}. "
        f"Set pass=true when score >= {pass_threshold:.3f}."
    )


def build_judge_messages(example: NormalizedExample, model_output: str, pass_threshold: float) -> List[LLMMessage]:
    rubric_text = _rubric_for_prompt(example.rubric)
    reference = example.reference_answer or ""

    system = (
        "You are a strict LLM judge. Return only valid JSON and no markdown. "
        "The JSON schema is: "
        "{\"score\": float, \"pass\": bool, \"rationale\": str, \"criteria\": {\"name\": float}}. "
        "score must be between 0 and 1."
    )

    user = (
        f"Provenance: {example.provenance}\n"
        f"Judge mode: {example.judge_mode}\n\n"
        f"Task instructions:\n{example.instructions}\n\n"
        f"Task context:\n{example.context or '(none)'}\n\n"
        f"Reference answer:\n{reference or '(none)'}\n\n"
        f"Rubric criteria:\n{rubric_text}\n\n"
        f"Candidate answer:\n{model_output}\n\n"
        f"{_judge_instruction(example, pass_threshold)}"
    )

    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


def build_rubric_criterion_judge_messages(
    example: NormalizedExample,
    model_output: str,
    criterion: Dict[str, Any],
    criterion_index: int,
    pass_threshold: float,
) -> List[LLMMessage]:
    reference = example.reference_answer or ""
    criterion_id = str(criterion.get("id", f"criterion_{criterion_index}")).strip()
    criterion_text = _rubric_criterion_for_prompt(criterion, criterion_index)

    system = (
        "You are a strict LLM judge. Return only valid JSON and no markdown. "
        "The JSON schema is: "
        "{\"grade\": 0|1, \"reasoning\": str, \"criterion_id\": str}. "
        "grade must be exactly 0 or 1."
    )

    user = (
        f"Provenance: {example.provenance}\n"
        "Judge mode: rubric\n\n"
        f"Task instructions:\n{example.instructions}\n\n"
        f"Task context:\n{example.context or '(none)'}\n\n"
        f"Reference answer:\n{reference or '(none)'}\n\n"
        "Evaluate only this single rubric criterion:\n"
        f"{criterion_text}\n\n"
        f"Candidate answer:\n{model_output}\n\n"
        f"Score only criterion '{criterion_id}'. "
        "Assign grade=1 only if the criterion is clearly satisfied; otherwise grade=0. "
        f"Set criterion_id exactly to {json.dumps(criterion_id)}. "
        "Return only this JSON object: "
        "{\"grade\": 0|1, \"reasoning\": str, \"criterion_id\": str}."
    )

    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


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

    if "score" in obj:
        score_raw = obj.get("score", 0.0)
    elif "grade" in obj:
        grade_raw = obj.get("grade", 0)
        if isinstance(grade_raw, bool):
            score_raw = 1.0 if grade_raw else 0.0
        elif isinstance(grade_raw, (int, float)):
            score_raw = 1.0 if float(grade_raw) >= 0.5 else 0.0
        elif isinstance(grade_raw, str):
            normalized = grade_raw.strip().lower()
            if normalized in {"1", "true", "yes", "met"}:
                score_raw = 1.0
            elif normalized in {"0", "false", "no", "not_met", "not met"}:
                score_raw = 0.0
            else:
                score_raw = 0.0
        else:
            score_raw = 0.0
    else:
        score_raw = 0.0

    score = float(score_raw)
    score = max(0.0, min(1.0, score))

    parsed_pass = obj.get("pass")
    if isinstance(parsed_pass, bool):
        passed = parsed_pass
    else:
        passed = score >= fallback_pass_threshold

    rationale = str(obj.get("rationale") or obj.get("reasoning") or "").strip()

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
    parsed: JudgeResult, example: NormalizedExample, pass_threshold: float
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

        weight = _criterion_weight(item)
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
    weighted_score = max(0.0, min(1.0, float(weighted_score)))

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
            },
        },
    )
