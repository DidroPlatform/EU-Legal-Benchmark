from __future__ import annotations

from typing import Any, Dict, List, Protocol

from src.types import JudgeResult, LLMMessage, NormalizedExample


class JudgePolicyHandler(Protocol):
    policy_id: str

    def build_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        pass_threshold: float,
    ) -> List[LLMMessage]: ...

    def build_rubric_criterion_judge_messages(
        self,
        example: NormalizedExample,
        model_output: str,
        criterion: Dict[str, Any],
        criterion_index: int,
        pass_threshold: float,
    ) -> List[LLMMessage]: ...

    def postprocess_judge_result(
        self,
        result: JudgeResult,
        example: NormalizedExample,
        pass_threshold: float,
    ) -> JudgeResult: ...
