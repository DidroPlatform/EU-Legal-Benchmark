from __future__ import annotations

from typing import Any, Dict, List

from src.types import JudgeResult, LLMMessage, NormalizedExample

from .shared import (
    clean_model_output_for_judge,
    conversation_for_judge,
    judge_instruction,
    rubric_criterion_for_prompt,
    rubric_for_prompt,
)


class DefaultJudgePolicyHandler:
    policy_id = "default_v1"

    def build_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        rubric_text = rubric_for_prompt(example.rubric)
        reference = example.reference_answer or ""

        system = (
            "You are a strict LLM judge. Return only valid JSON and no markdown. "
            "The JSON schema is: "
            '{"score": float, "pass": bool, "rationale": str, "criteria": {"name": float}}. '
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
            f"{judge_instruction(example, pass_threshold)}"
        )

        return [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ]

    def build_rubric_criterion_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        criterion: Dict[str, Any],
        criterion_index: int,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        del pass_threshold  # prompt is deterministic regardless of threshold
        criterion_id = str(criterion.get("id", f"criterion_{criterion_index}")).strip()
        conversation = conversation_for_judge(example, model_output)
        cleaned_output = clean_model_output_for_judge(example, model_output)
        reference = example.reference_answer or ""
        criterion_text = rubric_criterion_for_prompt(criterion, criterion_index)

        system = (
            "You are a strict LLM judge. Return only valid JSON and no markdown. "
            "The JSON schema is: "
            '{"criteria_met": bool, "explanation": str}. '
            "criteria_met must be true or false."
        )

        user = (
            f"Provenance: {example.provenance}\n"
            "Judge mode: rubric\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Reference answer:\n{reference or '(none)'}\n\n"
            "Evaluate only this single rubric criterion:\n"
            f"{criterion_text}\n\n"
            f"Candidate answer:\n{cleaned_output}\n\n"
            f"Score only criterion '{criterion_id}'. "
            "Set criteria_met=true only if the criterion is clearly satisfied; otherwise false. "
            "Return only this JSON object: "
            '{"criteria_met": bool, "explanation": str}.'
        )

        return [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ]

    def postprocess_judge_result(
        self,
        result: JudgeResult,
        example: NormalizedExample,
        pass_threshold: float,
    ) -> JudgeResult:
        del example, pass_threshold
        return result


DEFAULT_POLICY_HANDLER = DefaultJudgePolicyHandler()
