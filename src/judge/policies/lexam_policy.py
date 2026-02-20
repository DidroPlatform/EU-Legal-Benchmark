from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List

from src.types import JudgeResult, LLMMessage, NormalizedExample, clamp_score_01

from .default_policy import DEFAULT_POLICY_HANDLER


LEXAM_JUDGE_SYSTEM = "Act as a Judge specializing in the evaluation of Swiss law schools exams. Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning."

LEXAM_JUDGE_USER_PROMPT = """Goal:
Your task is to assess how well the response aligns with the reference answer, with a focus on accuracy, completeness, and legal reasoning.

Context:
You will be provided with a response (labeled: Model's Answer) to a law school exam question (labeled: Question) and a reference answer (labeled: Reference Answer). 

Return format:
    After reviewing the response:
    1. Explanation: Briefly explain your reasoning regarding how the response conforms to or deviates from the reference answer. 
    2. Constructive feedback: Additionally, provide neutral, constructive feedback and corrections in the style of a university professor.
    3. Correctness score: Assign a final correctness score on a scale from 0.0 to 1.0 (in increments of 0.1). This score should reflect the extent to which the response satisfies the reference answer, where 
        - 1.0 = complete fulfillment (100%) 
        - lower scores reflect proportionate shortfalls (e.g. 0.5 = 50% fulfillment). 
        - The correctness score will be provided in the JSON output format specified below.

Warnings:
    - In some cases, the reference answer may include only keywords or factual elements to be examined, along with (+), (-) or (+/-). Respect these indications when determining correctness:
        - (+) means the element must be affirmed.
        - (–) means the element must be denied.
        - (-/+) indicates that arguments in either direction are acceptable if legally sound.
    - Deviations or additional elements not found in the reference answer should generally be penalized unless you are certain they are legally correct and relevant. Assume the reference answer includes all information necessary for a perfect response.
    - The reference answer may contain citations (e.g., from books or law review articles), which the response does not need to replicate. However, statutes should be cited precisely, specifying Abs., Ziff., or lit. whenever applicable.
    - If the reference answer includes separate sub-points, use these for proportional scoring guidance (e.g., addressing 2 out of 4 sub-points correctly equals approximately a 0.5 score).
Judge the below case, give the brief reasoning process and the final grade.
"""

LEXAM_JUDGE_JSON_OUTPUT_INSTRUCTION = """Return only valid JSON (no markdown) with exactly this schema:
{"score": <float 0.0-1.0 step 0.1>, "rationale": "<brief explanation>", "constructive_feedback": "<neutral professor-style feedback>", "criteria": {"overall": <same score>}, "pass": <bool>}"""


class LEXamJudgePolicyHandler:
    policy_id = "lexam_oq_v1"

    def build_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        pass_threshold: float,
    ) -> List[LLMMessage]:
        if example.judge_mode != "reference":
            return DEFAULT_POLICY_HANDLER.build_judge_messages(example, model_output, pass_threshold)

        reference = example.reference_answer or ""
        user = (
            f"{LEXAM_JUDGE_USER_PROMPT}\n\n"
            f"{LEXAM_JUDGE_JSON_OUTPUT_INSTRUCTION}\n\n"
            "Question:\n"
            f"```{example.instructions}```\n\n"
            "Reference Answer:\n"
            f"```{reference}```\n\n"
            "Model's Answer:\n"
            f"```[{model_output}]```\n\n"
            "Your Judgment:\n"
        )
        return [
            LLMMessage(role="system", content=LEXAM_JUDGE_SYSTEM),
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
        return DEFAULT_POLICY_HANDLER.build_rubric_criterion_judge_messages(
            example,
            model_output,
            criterion,
            criterion_index,
            pass_threshold,
        )

    def postprocess_judge_result(
        self,
        result: JudgeResult,
        example: NormalizedExample,
        pass_threshold: float,
    ) -> JudgeResult:
        if example.judge_mode != "reference":
            return result

        quantized = float(
            Decimal(str(result.score)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        )
        quantized = clamp_score_01(quantized)

        criteria = result.criteria if isinstance(result.criteria, dict) else {}
        if not criteria or set(criteria.keys()) == {"overall"}:
            criteria = {"overall": quantized}

        return replace(
            result,
            score=quantized,
            passed=quantized >= pass_threshold,
            criteria=criteria,
        )


LEXAM_POLICY_HANDLER = LEXamJudgePolicyHandler()
