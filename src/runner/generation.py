from __future__ import annotations

import concurrent.futures
import sys
from typing import Callable, Dict, List, Set, Tuple

from src.config import BenchmarkConfig, ModelConfig
from src.runner.contracts import (
    FailureItem,
    GenerationExecutionResult,
    GenerationItemResult,
    GenerationPhaseResult,
    GenerationTask,
    ModelCallPayload,
)
from src.runner.context import RunnerContext
from src.runner.response_sources import load_prefilled_responses, load_previous_output_responses
from src.runner.row_types import ResponseRow, TraceRow
from src.runner.row_builders import build_generation_trace, build_response_row
from src.types import FinalResponseSource, LLMMessage, NormalizedExample, WaitableRateLimiter


def _is_candidate_fatal_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "on-demand throughput isn’t supported" in message
        or "on-demand throughput isn't supported" in message
        or ("inference profile" in message and "bedrockexception" in message)
    )


def _plan_generation_tasks(
    config: BenchmarkConfig,
    examples: List[NormalizedExample],
) -> Tuple[List[GenerationTask], int]:
    total_items = len(config.candidates) * len(examples)
    generation_tasks: List[GenerationTask] = []
    display_index = 0
    for candidate in config.candidates:
        for example in examples:
            display_index += 1
            generation_tasks.append((display_index, candidate, example))
    return generation_tasks, total_items


def _resolve_response_source(
    config: BenchmarkConfig,
    candidate_names: List[str],
) -> Tuple[FinalResponseSource, Dict[Tuple[str, str], str]]:
    response_source: FinalResponseSource = config.run.final_response_source
    responses_by_key: Dict[Tuple[str, str], str] = {}

    if response_source == "prefilled":
        prefilled_path = config.run.prefilled_responses_path
        if prefilled_path is None:
            raise ValueError("Config validation failure: missing prefilled responses path.")
        responses_by_key = load_prefilled_responses(prefilled_path)
    elif response_source == "part_of_conversation":
        previous_output_path = config.run.previous_output_path
        if previous_output_path is None:
            raise ValueError("Config validation failure: missing previous output path.")
        responses_by_key = load_previous_output_responses(
            path=previous_output_path,
            candidate_names=candidate_names,
        )
    elif response_source != "sampled":
        raise ValueError(f"Unsupported response source after validation: {response_source}")

    return response_source, responses_by_key


def _validate_preloaded_responses_coverage(
    response_source: FinalResponseSource,
    generation_tasks: List[GenerationTask],
    responses_by_key: Dict[Tuple[str, str], str],
) -> None:
    if response_source not in {"prefilled", "part_of_conversation"}:
        return

    missing = []
    for _, candidate, example in generation_tasks:
        key = (example.id, candidate.name)
        if key not in responses_by_key:
            missing.append(f"{example.id}:{candidate.name}")
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise ValueError(
            f"Missing {response_source} responses for selected tasks: " + preview + suffix
        )


def _generate_candidate_response(
    *,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    total_items: int,
    ctx: RunnerContext,
    response_source: FinalResponseSource,
    responses_by_key: Dict[Tuple[str, str], str],
    build_candidate_messages: Callable[[NormalizedExample, str], List[LLMMessage]],
    response_rate_limiter: WaitableRateLimiter,
    provider_response_rate_limiters: Dict[str, WaitableRateLimiter],
) -> GenerationItemResult:
    provider = ctx.providers.get(candidate.provider)
    candidate_messages = build_candidate_messages(example, ctx.config.run.default_system_prompt)
    if ctx.config.run.use_scratchpad:
        scratchpad = str(example.metadata.get("scratchpad", "")).strip()
        if scratchpad:
            candidate_messages = [
                *candidate_messages,
                LLMMessage(role="user", content=f"Scratchpad:\n{scratchpad}"),
            ]

    response_req_id = ctx.request_id(ctx.run_id, candidate.name, example.id, stage="response")
    request = ctx.build_request(candidate, candidate_messages, response_req_id)
    request.response_api = ctx.config.run.response_api
    if ctx.config.run.web_search:
        existing_extra = dict(request.extra_body or {})
        existing_extra.setdefault("web_search", True)
        request.extra_body = existing_extra

    def _wait_for_generation_slot(_: int) -> None:
        response_rate_limiter.wait()
        provider_limiter = provider_response_rate_limiters.get(candidate.provider)
        if provider_limiter is not None:
            provider_limiter.wait()

    ctx.emit_progress(
        ctx.progress_mode,
        ctx.progress_line(
            item=f"{display_index}/{total_items}",
            stage="response_started",
            candidate=candidate.name,
            dataset=example.dataset_name,
            example=example.id,
        ),
    )

    if response_source in {"prefilled", "part_of_conversation"}:
        response_payload: ModelCallPayload = {
            "provider": response_source,
            "model": candidate.model,
            "text": responses_by_key[(example.id, candidate.name)],
            "usage": {},
            "latency_s": None,
            "request_id": response_req_id,
        }
        cache_hit = False
        cache_key = None
    else:
        if provider is None:
            raise ValueError(
                f"Missing initialized provider '{candidate.provider}' for candidate '{candidate.name}'."
            )
        response_payload, cache_hit, cache_key = ctx.run_model_call(
            provider,
            request,
            ctx.cache,
            ctx.config.retry,
            stage="response",
            include_raw=ctx.config.run.include_raw_provider_response,
            before_attempt=_wait_for_generation_slot,
        )

    response_row = build_response_row(
        run_id=ctx.run_id,
        run_started_at_utc=ctx.run_started_at_utc,
        example=example,
        candidate=candidate,
        response_payload=response_payload,
        response_req_id=response_req_id,
        cache_key=cache_key,
        cache_hit=cache_hit,
        response_source=response_source,
        candidate_messages=candidate_messages,
        to_jsonable_messages=ctx.to_jsonable_messages,
        model_settings=ctx.model_settings,
    )

    generation_trace = build_generation_trace(
        run_id=ctx.run_id,
        example=example,
        candidate=candidate,
        response_row=response_row,
        candidate_messages=candidate_messages,
        cache_key=cache_key,
        cache_hit=cache_hit,
        request_response_api=request.response_api,
        request_extra_body=request.extra_body,
        to_jsonable_messages=ctx.to_jsonable_messages,
    )

    return GenerationItemResult(
        display_index=display_index,
        candidate=candidate,
        example=example,
        response_row=response_row,
        cache_hit=cache_hit,
        generation_trace=generation_trace,
    )


def _execute_generation_workers(
    *,
    generation_tasks: List[GenerationTask],
    response_workers: int,
    response_builder: Callable[[int, ModelConfig, NormalizedExample], GenerationItemResult],
    total_items: int,
    ctx: RunnerContext,
) -> GenerationExecutionResult:
    generation_results: List[GenerationItemResult] = []
    failed_items: List[FailureItem] = []
    interrupted = False
    generation_worker_count = min(response_workers, len(generation_tasks))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=generation_worker_count)
    inflight: Dict[concurrent.futures.Future, Tuple[int, ModelConfig, NormalizedExample]] = {}
    blocked_candidates: Set[str] = set()
    next_task_idx = 0

    def _mark_skipped_task(
        display_index: int,
        candidate: ModelConfig,
        example: NormalizedExample,
        reason: str,
    ) -> None:
        ctx.emit_progress(
            ctx.progress_mode,
            ctx.progress_line(
                item=f"{display_index}/{total_items}",
                stage="response_skipped",
                candidate=candidate.name,
                dataset=example.dataset_name,
                example=example.id,
                reason=reason,
            ),
        )
        failed_items.append(
            _build_generation_failure_item(
                display_index=display_index,
                candidate=candidate,
                example=example,
                reason=reason,
            )
        )

    def _submit_until_full() -> None:
        nonlocal next_task_idx
        while len(inflight) < generation_worker_count and next_task_idx < len(generation_tasks):
            display_index, candidate, example = generation_tasks[next_task_idx]
            next_task_idx += 1
            _submit_generation_task(
                blocked_candidates=blocked_candidates,
                pool=pool,
                inflight=inflight,
                display_index=display_index,
                candidate=candidate,
                example=example,
                total_items=total_items,
                ctx=ctx,
                response_builder=response_builder,
                mark_skipped_task=_mark_skipped_task,
            )

    try:
        _submit_until_full()
        while inflight:
            done, _ = concurrent.futures.wait(
                set(inflight.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                _handle_generation_future(
                    future=future,
                    inflight=inflight,
                    generation_results=generation_results,
                    failed_items=failed_items,
                    blocked_candidates=blocked_candidates,
                )
                _submit_until_full()
    except KeyboardInterrupt:
        interrupted = True
        ctx.emit_progress(
            ctx.progress_mode,
            ctx.progress_line(
                stage="response_phase_interrupted",
                total_items=total_items,
                completed_items=len(generation_results) + len(failed_items),
            ),
        )
        for future in inflight:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
    else:
        pool.shutdown(wait=True, cancel_futures=False)

    return GenerationExecutionResult(
        generation_results=generation_results,
        failed_items=failed_items,
        interrupted=interrupted,
    )


def _build_generation_failure_item(
    *,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    reason: str,
) -> FailureItem:
    return {
        "display_index": display_index,
        "candidate_name": candidate.name,
        "example_id": example.id,
        "dataset": example.dataset_name,
        "error": reason,
        "stage": "response",
        "judge_mode": example.judge_mode,
    }


def _submit_generation_task(
    *,
    blocked_candidates: Set[str],
    pool: concurrent.futures.ThreadPoolExecutor,
    inflight: Dict[concurrent.futures.Future, Tuple[int, ModelConfig, NormalizedExample]],
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    total_items: int,
    ctx: RunnerContext,
    response_builder: Callable[[int, ModelConfig, NormalizedExample], GenerationItemResult],
    mark_skipped_task: Callable[[int, ModelConfig, NormalizedExample, str], None],
) -> None:
    if candidate.name in blocked_candidates:
        mark_skipped_task(
            display_index,
            candidate,
            example,
            "Skipped due to earlier fatal provider error for this candidate.",
        )
        return

    ctx.emit_progress(
        ctx.progress_mode,
        ctx.progress_line(
            item=f"{display_index}/{total_items}",
            stage="response_queued",
            candidate=candidate.name,
            dataset=example.dataset_name,
            example=example.id,
        ),
    )
    fut = pool.submit(response_builder, display_index, candidate, example)
    inflight[fut] = (display_index, candidate, example)


def _handle_generation_future(
    *,
    future: concurrent.futures.Future,
    inflight: Dict[concurrent.futures.Future, Tuple[int, ModelConfig, NormalizedExample]],
    generation_results: List[GenerationItemResult],
    failed_items: List[FailureItem],
    blocked_candidates: Set[str],
) -> None:
    di, cand, ex = inflight.pop(future)
    try:
        generation_results.append(future.result())
    except Exception as exc:
        print(
            f"[error] generation failed for candidate={cand.name} "
            f"dataset={ex.dataset_name} example={ex.id}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        failed_items.append(
            _build_generation_failure_item(
                display_index=di,
                candidate=cand,
                example=ex,
                reason=str(exc),
            )
        )
        if _is_candidate_fatal_error(exc) and cand.name not in blocked_candidates:
            blocked_candidates.add(cand.name)
            print(
                f"[warning] disabling remaining tasks for candidate={cand.name} "
                "after fatal provider configuration error",
                file=sys.stderr,
                flush=True,
            )


def _build_generation_artifacts(
    *,
    generation_results: List[GenerationItemResult],
    total_items: int,
    ctx: RunnerContext,
) -> Tuple[List[ResponseRow], List[TraceRow]]:
    responses_rows: List[ResponseRow] = []
    trace_rows: List[TraceRow] = []

    for generation_result in generation_results:
        display_index = generation_result.display_index
        candidate = generation_result.candidate
        example = generation_result.example
        response_row = generation_result.response_row
        cache_hit = generation_result.cache_hit

        responses_rows.append(response_row)
        ctx.emit_progress(
            ctx.progress_mode,
            ctx.progress_line(
                item=f"{display_index}/{total_items}",
                stage="response_done",
                candidate=candidate.name,
                dataset=example.dataset_name,
                example=example.id,
                cache_hit=cache_hit,
                latency_s=response_row["latency_s"],
            ),
        )
        trace_rows.append(generation_result.generation_trace)

    return responses_rows, trace_rows


def run_generation_phase(
    *,
    ctx: RunnerContext,
    examples: List[NormalizedExample],
    response_workers: int,
    response_rate_limiter: WaitableRateLimiter,
    provider_response_rate_limiters: Dict[str, WaitableRateLimiter],
    build_candidate_messages: Callable[[NormalizedExample, str], List[LLMMessage]],
) -> GenerationPhaseResult:
    generation_tasks, total_items = _plan_generation_tasks(ctx.config, examples)
    response_source, responses_by_key = _resolve_response_source(
        ctx.config,
        candidate_names=[candidate.name for candidate in ctx.config.candidates],
    )
    _validate_preloaded_responses_coverage(response_source, generation_tasks, responses_by_key)

    ctx.emit_progress(
        ctx.progress_mode,
        ctx.progress_line(
            stage="response_phase_start",
            total_items=total_items,
            workers=min(response_workers, max(1, len(generation_tasks))),
            rpm=ctx.config.run.response_rate_limit_rpm,
        ),
    )

    execution = _execute_generation_workers(
        generation_tasks=generation_tasks,
        response_workers=response_workers,
        response_builder=lambda display_index, candidate, example: _generate_candidate_response(
            display_index=display_index,
            candidate=candidate,
            example=example,
            total_items=total_items,
            ctx=ctx,
            response_source=response_source,
            responses_by_key=responses_by_key,
            build_candidate_messages=build_candidate_messages,
            response_rate_limiter=response_rate_limiter,
            provider_response_rate_limiters=provider_response_rate_limiters,
        ),
        total_items=total_items,
        ctx=ctx,
    )

    generation_results = execution.generation_results
    generation_results.sort(key=lambda item: item.display_index)

    responses_rows, trace_rows = _build_generation_artifacts(
        generation_results=generation_results,
        total_items=total_items,
        ctx=ctx,
    )

    interrupted = execution.interrupted
    if not interrupted:
        ctx.emit_progress(ctx.progress_mode, ctx.progress_line(stage="response_phase_done", total_items=total_items))

    return GenerationPhaseResult(
        generation_results=generation_results,
        responses_rows=responses_rows,
        trace_rows=trace_rows,
        failed_items=execution.failed_items,
        interrupted=interrupted,
        interrupted_stage="generation" if interrupted else None,
    )
