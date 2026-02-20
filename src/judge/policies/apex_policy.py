from __future__ import annotations

from typing import Any, Dict, List

from src.types import JudgeResult, LLMMessage, NormalizedExample

from .default_policy import DEFAULT_POLICY_HANDLER
from .shared import clean_model_output_for_judge


APEX_V1_GRADING_PROMPT_TEMPLATE = """You are evaluating a model-generated response
against a specific criterion. Your task is to
determine if the response satisfies this criterion and provide a concise explanation.

Criterion to evaluate: {criterion_description}

Response to evaluate: {solution}

Instructions:
1. First, analyze the response against the given criterion.
2. Determine if the response fully satisfies the criterion (result = 1) or not (result = 0).
3. Provide a concise explanation (maximum 2-3 sentences) that:
    a. States whether the criterion is met or not
    b. Points to specific evidence from the response
    c. Avoids unnecessary details or repetition

Return your evaluation in the following JSON format:
{{
    "result": <1 or 0>,
    "reason": "<concise explanation>"
}}

Keep your explanation brief and focus on the key points that justify your result.
"""


class ApexJudgePolicyHandler:
    policy_id = "apexv1_extended_v1"

    def build_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        return DEFAULT_POLICY_HANDLER.build_judge_messages(example, model_output, pass_threshold)

    def build_rubric_criterion_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        criterion: Dict[str, Any],
        criterion_index: int,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        del criterion_index, pass_threshold
        criterion_id = str(criterion.get("id", "criterion")).strip() or "criterion"
        cleaned_output = clean_model_output_for_judge(example, model_output)
        criterion_description = (
            str(criterion.get("description", "")).strip()
            or str(criterion.get("title", "")).strip()
            or criterion_id
        )
        user = APEX_V1_GRADING_PROMPT_TEMPLATE.format(
            criterion_description=criterion_description,
            solution=cleaned_output,
        )
        return [LLMMessage(role="user", content=user)]

    def postprocess_judge_result(
        self,
        result: JudgeResult,
        example: NormalizedExample,
        pass_threshold: float,
    ) -> JudgeResult:
        del example, pass_threshold
        return result


APEX_POLICY_HANDLER = ApexJudgePolicyHandler()
