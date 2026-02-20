from __future__ import annotations

import re
from typing import Any, Dict, List

from src.data.policies import get_policy
from src.types import NormalizedExample


def criterion_weight(item: Dict[str, Any]) -> float:
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


def rubric_for_prompt(rubric: List[Dict[str, Any]] | None) -> str:
    if not rubric:
        return "No rubric provided."

    lines = []
    for i, item in enumerate(rubric, start=1):
        title = str(item.get("title", f"Criterion {i}"))
        criterion_id = str(item.get("id", f"criterion_{i}"))
        weight = criterion_weight(item)
        lines.append(f"- {criterion_id}: {title} (weight_hint={weight})")
    return "\n".join(lines)


def rubric_criterion_for_prompt(item: Dict[str, Any], criterion_index: int) -> str:
    title = str(item.get("title", f"Criterion {criterion_index}"))
    criterion_id = str(item.get("id", f"criterion_{criterion_index}"))
    weight = criterion_weight(item)
    return f"- {criterion_id}: {title} (weight_hint={weight})"


def clean_model_output_for_judge(example: NormalizedExample, text: str) -> str:
    policy = get_policy(example.metadata.get("policy_id"))
    if policy.policy_id != "prbench_v1":
        return text
    cleaned = re.sub(
        r"<(?:think|thinking|reasoning|analysis)\\b[^>]*>.*?</(?:think|thinking|reasoning|analysis)>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def conversation_for_judge(example: NormalizedExample, model_output: str) -> str:
    cleaned_output = clean_model_output_for_judge(example, model_output)
    if example.messages:
        lines = [f"{msg.role}: {msg.content}" for msg in example.messages if msg.content.strip()]
        lines.append(f"assistant: {cleaned_output}")
        return "\n".join(lines)

    fallback_parts = [f"user: {example.instructions}"]
    if example.context:
        fallback_parts.append(f"context: {example.context}")
    fallback_parts.append(f"assistant: {cleaned_output}")
    return "\n".join(fallback_parts)


def judge_instruction(example: NormalizedExample, pass_threshold: float) -> str:
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
