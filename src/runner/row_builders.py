from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.config import ModelConfig
from src.types import JudgeResult, LLMMessage, NormalizedExample
from src.runner.contracts import ModelCallPayload
from src.runner.row_types import JudgmentRow, ResponseRow, TraceRow


def build_response_row(
    *,
    run_id: str,
    run_started_at_utc: str,
    example: NormalizedExample,
    candidate: ModelConfig,
    response_payload: ModelCallPayload,
    response_req_id: str,
    cache_key: Optional[str],
    cache_hit: bool,
    response_source: str,
    candidate_messages: List[LLMMessage],
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]],
    model_settings: Callable[[ModelConfig], Dict[str, Any]],
) -> ResponseRow:
    return {
        "run_id": run_id,
        "run_started_at_utc": run_started_at_utc,
        "dataset": example.dataset_name,
        "provenance": example.provenance,
        "judge_mode": example.judge_mode,
        "example_id": example.id,
        "candidate_name": candidate.name,
        "candidate_provider": candidate.provider,
        "candidate_model": candidate.model,
        "candidate_settings": model_settings(candidate),
        "request_id": response_payload.get("request_id") or response_req_id,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "response_source": response_source,
        "prompt_messages": to_jsonable_messages(candidate_messages),
        "response_text": response_payload.get("text", ""),
        "usage": response_payload.get("usage", {}),
        "latency_s": response_payload.get("latency_s"),
        "metadata": example.metadata,
        "reference_answer": example.reference_answer,
    }


def build_generation_trace(
    *,
    run_id: str,
    example: NormalizedExample,
    candidate: ModelConfig,
    response_row: ResponseRow,
    candidate_messages: List[LLMMessage],
    cache_key: Optional[str],
    cache_hit: bool,
    request_response_api: str,
    request_extra_body: Optional[Dict[str, Any]],
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]],
) -> TraceRow:
    return {
        "run_id": run_id,
        "event": "generation_call",
        "dataset": example.dataset_name,
        "example_id": example.id,
        "model_name": candidate.name,
        "provider": candidate.provider,
        "request_id": response_row["request_id"],
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "request": {
            "messages": to_jsonable_messages(candidate_messages),
            "temperature": candidate.temperature,
            "top_p": candidate.top_p,
            "frequency_penalty": candidate.frequency_penalty,
            "presence_penalty": candidate.presence_penalty,
            "max_tokens": candidate.max_tokens,
            "seed": candidate.seed,
            "response_api": request_response_api,
            "reasoning_effort": candidate.reasoning_effort,
            "thinking_budget": candidate.thinking_budget,
            "extra_body": request_extra_body,
        },
        "response": {
            "text": response_row["response_text"],
            "usage": response_row["usage"],
            "latency_s": response_row["latency_s"],
        },
    }


def build_judgment_row(
    *,
    run_id: str,
    run_started_at_utc: str,
    example: NormalizedExample,
    candidate_name: str,
    judge_name: str,
    judge_provider: str,
    judge_model: str,
    judge_settings: Dict[str, Any],
    request_id: str,
    cache_key: Optional[str],
    cache_hit: bool,
    result: JudgeResult,
) -> JudgmentRow:
    aggregation: Dict[str, Any] = {}
    if isinstance(result.raw, dict):
        aggregation = result.raw.get("deterministic_rubric_aggregation", {}) or {}
    return {
        "run_id": run_id,
        "run_started_at_utc": run_started_at_utc,
        "dataset": example.dataset_name,
        "provenance": example.provenance,
        "judge_mode": example.judge_mode,
        "example_id": example.id,
        "candidate_name": candidate_name,
        "judge_name": judge_name,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_settings": judge_settings,
        "request_id": request_id,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "score": result.score,
        "pass": result.passed,
        "rationale": result.rationale,
        "criteria": result.criteria,
        "parse_error": result.parse_error,
        "prbench_weighted_raw": aggregation.get("raw_sum"),
        "prbench_points_normalized": aggregation.get("normalized_points"),
        "prbench_points_clipped": aggregation.get("clipped_points"),
        "raw_judge": result.raw,
    }


def build_judge_trace_request(
    *,
    model: ModelConfig,
    messages: List[LLMMessage],
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]],
) -> Dict[str, Any]:
    return {
        "messages": to_jsonable_messages(messages),
        "temperature": model.temperature,
        "top_p": model.top_p,
        "frequency_penalty": model.frequency_penalty,
        "presence_penalty": model.presence_penalty,
        "max_tokens": model.max_tokens,
        "seed": model.seed,
        "extra_body": model.extra_body,
    }


def build_judge_trace(
    *,
    run_id: str,
    example: NormalizedExample,
    judge_model: ModelConfig,
    request_id: str,
    cache_key: Optional[str],
    cache_hit: bool,
    judge_messages: List[LLMMessage],
    judge_payload: ModelCallPayload,
    parsed: JudgeResult,
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]],
    criterion_id: Optional[str] = None,
    criterion_index: Optional[int] = None,
    criterion_score: Optional[float] = None,
    error: Optional[str] = None,
) -> TraceRow:
    row: TraceRow = {
        "run_id": run_id,
        "event": "judge_call",
        "dataset": example.dataset_name,
        "example_id": example.id,
        "model_name": judge_model.name,
        "provider": judge_model.provider,
        "request_id": request_id,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "request": build_judge_trace_request(
            model=judge_model,
            messages=judge_messages,
            to_jsonable_messages=to_jsonable_messages,
        ),
        "response": {
            "text": judge_payload.get("text", ""),
            "usage": judge_payload.get("usage", {}),
            "latency_s": judge_payload.get("latency_s"),
            "parse_error": parsed.parse_error,
            "error": error if error is not None else judge_payload.get("error"),
        },
    }
    if criterion_id is not None:
        row["criterion_id"] = criterion_id
    if criterion_index is not None:
        row["criterion_index"] = criterion_index
    if criterion_score is not None:
        row["response"]["criterion_score"] = criterion_score
    return row


def build_mcq_trace(
    *,
    run_id: str,
    example: NormalizedExample,
    request_id: str,
    response_text: str,
    parsed: JudgeResult,
) -> TraceRow:
    return {
        "run_id": run_id,
        "event": "mcq_grade",
        "dataset": example.dataset_name,
        "example_id": example.id,
        "model_name": "deterministic_mcq",
        "provider": "programmatic",
        "request_id": request_id,
        "cache_key": None,
        "cache_hit": False,
        "request": {
            "expected_choice_ids": example.metadata.get("correct_choice_ids", []),
        },
        "response": {
            "candidate_text": response_text,
            "parse_error": parsed.parse_error,
            "score": parsed.score,
            "criteria": parsed.criteria,
        },
    }
