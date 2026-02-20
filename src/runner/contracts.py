from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from src.config import ModelConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample

from .row_types import JudgmentRow, ResponseRow, TraceRow


class ModelCallPayload(TypedDict, total=False):
    provider: str
    model: str
    text: str
    usage: Dict[str, Any]
    latency_s: Optional[float]
    request_id: str
    raw_response: Optional[Dict[str, Any]]
    error: str


class FailureItem(TypedDict, total=False):
    display_index: int
    candidate_name: str
    example_id: str
    dataset: str
    error: str
    stage: str
    judge_mode: str
    criterion_id: str
    judge_provider: str
    judge_model: str
    request_id: str


@dataclass
class GenerationItemResult:
    display_index: int
    candidate: ModelConfig
    example: NormalizedExample
    response_row: ResponseRow
    cache_hit: bool
    generation_trace: TraceRow


@dataclass
class GenerationExecutionResult:
    generation_results: List[GenerationItemResult]
    failed_items: List[FailureItem]
    interrupted: bool


@dataclass
class GenerationPhaseResult:
    generation_results: List[GenerationItemResult]
    responses_rows: List[ResponseRow]
    trace_rows: List[TraceRow]
    failed_items: List[FailureItem]
    interrupted: bool
    interrupted_stage: Optional[str]


@dataclass
class JudgeCallResult:
    judge_payload: ModelCallPayload
    judge_cache_hit: bool
    judge_cache_key: Optional[str]
    parsed: JudgeResult
    error: Optional[str]


@dataclass
class JudgeItemResult:
    judgment_row: JudgmentRow
    trace_rows: List[TraceRow]
    failed_items: List[FailureItem]


@dataclass
class JudgingPhaseResult:
    judgments_rows: List[JudgmentRow]
    trace_rows: List[TraceRow]
    failed_items: List[FailureItem]
    interrupted: bool
    interrupted_stage: Optional[str]


GenerationTask = Tuple[int, ModelConfig, NormalizedExample]


@dataclass
class RubricCriterionJudgeResult:
    criterion_id: str
    criterion_index: int
    judge_model: ModelConfig
    judge_messages: List[LLMMessage]
    judge_req_id: str
    judge_payload: ModelCallPayload
    judge_cache_hit: bool
    judge_cache_key: Optional[str]
    parsed: JudgeResult
    criterion_score: float
    error: Optional[str]
