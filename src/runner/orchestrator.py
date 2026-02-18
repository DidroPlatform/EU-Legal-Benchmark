from __future__ import annotations

import datetime as dt
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.cache import DiskCache
from src.config import BenchmarkConfig, ModelConfig
from src.providers.base import GOOGLE_PROVIDER_NAMES
from src.types import LLMMessage, LLMRequest, NormalizedExample

from .generation import run_generation_phase
from .helpers import (
    _build_request,
    _emit_progress,
    _judge_descriptor,
    _model_settings,
    _progress_line,
    _request_id,
    _to_jsonable_messages,
)
from .judging import run_judge_phase
from .output import write_run_outputs
from .rate_limiter import PerMinuteRateLimiter


def run(
    config: BenchmarkConfig,
    limit_override: int | None = None,
    progress_mode: str = "log",
    *,
    validate_canonical_inputs: Callable[[BenchmarkConfig], None],
    load_all_examples: Callable[[BenchmarkConfig], tuple[List[NormalizedExample], List[Dict[str, Any]]]],
    required_provider_names: Callable[[BenchmarkConfig], set[str]],
    build_provider: Callable[[str, BenchmarkConfig], Any],
    run_model_call: Callable[..., Any],
    grade_mcq_output: Callable[..., Any],
    build_candidate_messages: Callable[[NormalizedExample, str], List[LLMMessage]],
    build_judge_messages: Callable[..., List[LLMMessage]],
    build_rubric_criterion_judge_messages: Callable[..., List[LLMMessage]],
    parse_judge_output: Callable[..., Any],
    resolve_rubric_criterion_score: Callable[..., Any],
    apply_weighted_rubric_score: Callable[..., Any],
    per_minute_rate_limiter_cls: type[PerMinuteRateLimiter] = PerMinuteRateLimiter,
    emit_progress: Callable[[str, str], None] = _emit_progress,
    progress_line: Callable[..., str] = _progress_line,
    request_id: Callable[[str, str, str, str], str] = _request_id,
    to_jsonable_messages: Callable[[List[LLMMessage]], List[Dict[str, str]]] = _to_jsonable_messages,
    build_request: Callable[[ModelConfig, List[LLMMessage], str], LLMRequest] = _build_request,
    model_settings: Callable[[ModelConfig], Dict[str, Any]] = _model_settings,
    judge_descriptor: Callable[[ModelConfig], Dict[str, Any]] = _judge_descriptor,
) -> Path:
    run_started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = config.run.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    )
    validate_canonical_inputs(config)

    examples, dataset_stats = load_all_examples(config)
    if limit_override is not None:
        examples = examples[: max(0, limit_override)]

    if not examples:
        raise ValueError("No examples selected after dataset filtering.")

    run_dir = Path(config.run.runs_root) / run_id
    output_dir = run_dir / Path(config.run.output_dir).name
    cache_dir = run_dir / Path(config.cache.dir).name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = DiskCache(str(cache_dir), enabled=config.cache.enabled)
    required_providers = required_provider_names(config)
    providers = {name: build_provider(name, config) for name in required_providers}
    judge_uses_google = any(j.provider in GOOGLE_PROVIDER_NAMES for j in config.judges)
    response_workers = max(1, int(config.run.response_parallel_workers))
    response_rate_limiter = per_minute_rate_limiter_cls(config.run.response_rate_limit_rpm)
    provider_response_rate_limiters = {
        provider_name: per_minute_rate_limiter_cls(rpm)
        for provider_name, rpm in config.run.provider_response_rate_limit_rpm.items()
    }
    judge_workers = max(1, int(config.run.judge_parallel_workers))
    judge_rate_limiter = (
        per_minute_rate_limiter_cls(config.run.judge_rate_limit_rpm)
        if judge_uses_google and config.run.judge_rate_limit_rpm > 0
        else None
    )
    total_items = len(config.candidates) * len(examples)

    try:
        emit_progress(
            progress_mode,
            progress_line(
                stage="start",
                run_id=run_id,
                examples=len(examples),
                candidates=len(config.candidates),
                total_items=total_items,
            ),
        )

        normalized_rows: List[Dict[str, Any]] = []
        for example in examples:
            normalized_rows.append(
                {
                    "example_id": example.id,
                    "dataset": example.dataset_name,
                    "provenance": example.provenance,
                    "judge_mode": example.judge_mode,
                    "instructions": example.instructions,
                    "context": example.context,
                    "reference_answer": example.reference_answer,
                    "metadata": example.metadata,
                }
            )

        generation = run_generation_phase(
            config=config,
            run_id=run_id,
            run_started_at_utc=run_started_at_utc,
            examples=examples,
            providers=providers,
            cache=cache,
            response_workers=response_workers,
            response_rate_limiter=response_rate_limiter,
            provider_response_rate_limiters=provider_response_rate_limiters,
            progress_mode=progress_mode,
            emit_progress=emit_progress,
            progress_line=progress_line,
            model_settings=model_settings,
            to_jsonable_messages=to_jsonable_messages,
            request_id=request_id,
            build_request=build_request,
            run_model_call=run_model_call,
            build_candidate_messages=build_candidate_messages,
        )

        if generation.get("interrupted"):
            judging = {"judgments_rows": [], "trace_rows": []}
        else:
            judging = run_judge_phase(
                config=config,
                run_id=run_id,
                run_started_at_utc=run_started_at_utc,
                total_items=total_items,
                generation_results=generation["generation_results"],
                providers=providers,
                cache=cache,
                judge_workers=judge_workers,
                judge_rate_limiter=judge_rate_limiter,
                progress_mode=progress_mode,
                emit_progress=emit_progress,
                progress_line=progress_line,
                request_id=request_id,
                build_request=build_request,
                model_settings=model_settings,
                judge_descriptor=judge_descriptor,
                to_jsonable_messages=to_jsonable_messages,
                run_model_call=run_model_call,
                grade_mcq_output=grade_mcq_output,
                build_judge_messages=build_judge_messages,
                build_rubric_criterion_judge_messages=build_rubric_criterion_judge_messages,
                parse_judge_output=parse_judge_output,
                resolve_rubric_criterion_score=resolve_rubric_criterion_score,
                apply_weighted_rubric_score=apply_weighted_rubric_score,
                google_provider_names=GOOGLE_PROVIDER_NAMES,
            )

        all_trace_rows = [*generation["trace_rows"], *judging["trace_rows"]]
        write_run_outputs(
            output_dir=output_dir,
            config=config,
            run_id=run_id,
            run_started_at_utc=run_started_at_utc,
            examples=examples,
            dataset_stats=dataset_stats,
            normalized_rows=normalized_rows,
            responses_rows=generation["responses_rows"],
            judgments_rows=judging["judgments_rows"],
            trace_rows=all_trace_rows,
            failed_items=generation["failed_items"],
            run_status="interrupted" if generation.get("interrupted") else "completed",
            interrupted_stage=generation.get("interrupted_stage"),
            judge_descriptor=judge_descriptor,
        )

        return output_dir
    finally:
        for provider in providers.values():
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()
