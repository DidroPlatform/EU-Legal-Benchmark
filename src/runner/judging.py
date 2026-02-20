from __future__ import annotations

import concurrent.futures
from dataclasses import replace
from typing import Any, Callable, Dict, List, Tuple

from src.cache import DiskCache
from src.config import BenchmarkConfig, ModelConfig
from src.providers.base import BaseProvider
from src.runner.context import RunnerContext
from src.runner.contracts import (
    FailureItem,
    GenerationItemResult,
    JudgeCallResult,
    JudgeItemResult,
    JudgingPhaseResult,
    ModelCallPayload,
    RubricCriterionJudgeResult,
)
from src.runner.helpers import build_error_judge_result
from src.runner.services import (
    ApplyPolicyScorePostprocessingFn,
    ApplyWeightedRubricScoreFn,
    BuildJudgeMessagesFn,
    BuildRequestFn,
    BuildRubricCriterionJudgeMessagesFn,
    GradeMcqOutputFn,
    JudgingServices,
    ParseJudgeOutputFn,
    ResolveRubricCriterionScoreFn,
    RunModelCallFn,
)
from src.runner.row_builders import build_judge_trace, build_judgment_row, build_mcq_trace
from src.runner.row_types import ResponseRow, TraceRow
from src.types import (
    JudgeResult,
    LLMMessage,
    NormalizedExample,
    WaitableRateLimiter,
)


def _enforce_fail_closed(result: JudgeResult) -> JudgeResult:
    if not result.parse_error:
        return result
    return replace(result, score=0.0, passed=False)


def _run_judge_call(
    *,
    judge_provider: BaseProvider,
    judge_model: ModelConfig,
    judge_req_id: str,
    judge_messages: List[LLMMessage],
    build_request: BuildRequestFn,
    run_model_call: RunModelCallFn,
    parse_judge_output: ParseJudgeOutputFn,
    cache: DiskCache,
    config: BenchmarkConfig,
    judge_rate_limiter: WaitableRateLimiter | None,
    google_provider_names: set[str],
    error_result_builder: Callable[[str], JudgeResult],
) -> JudgeCallResult:
    judge_request = build_request(judge_model, judge_messages, judge_req_id)
    judge_cache_hit = False
    judge_cache_key = None
    judge_payload: ModelCallPayload = {
        "text": "",
        "usage": {},
        "latency_s": None,
        "request_id": judge_req_id,
    }
    error_message: str | None = None

    try:
        if judge_rate_limiter is not None and judge_model.provider in google_provider_names:
            judge_rate_limiter.wait()
        judge_payload, judge_cache_hit, judge_cache_key = run_model_call(
            judge_provider,
            judge_request,
            cache,
            config.retry,
            stage="judge",
            include_raw=config.run.include_raw_provider_response,
        )
        parsed = parse_judge_output(
            raw_text=judge_payload.get("text", "{}"),
            fallback_pass_threshold=config.run.judge_pass_threshold,
        )
    except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary
        error_message = f"{type(exc).__name__}: {exc}"
        judge_payload["error"] = error_message
        parsed = error_result_builder(error_message)

    return JudgeCallResult(
        judge_payload=judge_payload,
        judge_cache_hit=judge_cache_hit,
        judge_cache_key=judge_cache_key,
        parsed=parsed,
        error=error_message,
    )


def _build_judge_failure_item(
    *,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    error: str,
    criterion_id: str | None = None,
    judge_model: ModelConfig | None = None,
    request_id: str | None = None,
) -> FailureItem:
    item: FailureItem = {
        "display_index": display_index,
        "candidate_name": candidate.name,
        "example_id": example.id,
        "dataset": example.dataset_name,
        "error": error,
        "stage": "judge",
        "judge_mode": example.judge_mode,
    }
    if criterion_id is not None:
        item["criterion_id"] = criterion_id
    if judge_model is not None:
        item["judge_provider"] = judge_model.provider
        item["judge_model"] = judge_model.model
    if request_id is not None:
        item["request_id"] = request_id
    return item


def _handle_mcq_judgment(
    *,
    ctx: RunnerContext,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    response_row: ResponseRow,
    grade_mcq_output: GradeMcqOutputFn,
) -> JudgeItemResult:
    parsed = grade_mcq_output(
        example=example,
        candidate_text=response_row["response_text"],
        pass_threshold=ctx.config.run.judge_pass_threshold,
    )
    parsed = _enforce_fail_closed(parsed)
    judge_req_id = ctx.request_id(
        ctx.run_id, "deterministic_mcq", f"{example.id}:{candidate.name}", stage="judge"
    )

    judgment_row = build_judgment_row(
        run_id=ctx.run_id,
        run_started_at_utc=ctx.run_started_at_utc,
        example=example,
        candidate_name=candidate.name,
        judge_name="deterministic_mcq",
        judge_provider="programmatic",
        judge_model="exact_match_v1",
        judge_settings={},
        request_id=judge_req_id,
        cache_key=None,
        cache_hit=False,
        result=parsed,
    )

    trace_row = build_mcq_trace(
        run_id=ctx.run_id,
        example=example,
        request_id=judgment_row["request_id"],
        response_text=response_row["response_text"],
        parsed=parsed,
    )

    return JudgeItemResult(
        judgment_row=judgment_row,
        trace_rows=[trace_row],
        failed_items=[],
    )


def _run_rubric_criterion(
    *,
    idx: int,
    criterion: Dict[str, Any],
    ctx: RunnerContext,
    candidate: ModelConfig,
    example: NormalizedExample,
    response_row: ResponseRow,
    rubric_judges: List[ModelConfig],
    judge_rate_limiter: WaitableRateLimiter | None,
    build_rubric_criterion_judge_messages: BuildRubricCriterionJudgeMessagesFn,
    parse_judge_output: ParseJudgeOutputFn,
    resolve_rubric_criterion_score: ResolveRubricCriterionScoreFn,
) -> RubricCriterionJudgeResult:
    criterion_id = str(criterion.get("id", f"criterion_{idx}")).strip()
    judge_model = rubric_judges[(idx - 1) % len(rubric_judges)]
    judge_provider = ctx.providers[judge_model.provider]
    judge_req_id = ctx.request_id(
        ctx.run_id,
        judge_model.name,
        f"{example.id}:{candidate.name}:{criterion_id}",
        stage="judge",
    )
    judge_messages = build_rubric_criterion_judge_messages(
        example=example,
        model_output=response_row["response_text"],
        criterion=criterion,
        criterion_index=idx,
        pass_threshold=ctx.config.run.judge_pass_threshold,
    )

    judge_call = _run_judge_call(
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_req_id=judge_req_id,
        judge_messages=judge_messages,
        build_request=ctx.build_request,
        run_model_call=ctx.run_model_call,
        parse_judge_output=parse_judge_output,
        cache=ctx.cache,
        config=ctx.config,
        judge_rate_limiter=judge_rate_limiter,
        google_provider_names=ctx.google_provider_names,
        error_result_builder=lambda error_message, _cid=criterion_id: build_error_judge_result(
            error_message, context=f"criterion '{_cid}'"
        ),
    )

    parsed = judge_call.parsed
    criterion_score, _ = resolve_rubric_criterion_score(
        criteria=parsed.criteria,
        criterion=criterion,
        criterion_index=idx,
        fallback_score=parsed.score,
    )

    return RubricCriterionJudgeResult(
        criterion_id=criterion_id,
        criterion_index=idx,
        judge_model=judge_model,
        judge_messages=judge_messages,
        judge_req_id=judge_req_id,
        judge_payload=judge_call.judge_payload,
        judge_cache_hit=judge_call.judge_cache_hit,
        judge_cache_key=judge_call.judge_cache_key,
        parsed=parsed,
        criterion_score=criterion_score,
        error=judge_call.error,
    )


def _collect_rubric_criterion_results(
    *,
    rubric_items: List[Dict[str, Any]],
    criterion_workers: int,
    ctx: RunnerContext,
    candidate: ModelConfig,
    example: NormalizedExample,
    response_row: ResponseRow,
    rubric_judges: List[ModelConfig],
    judge_rate_limiter: WaitableRateLimiter | None,
    build_rubric_criterion_judge_messages: BuildRubricCriterionJudgeMessagesFn,
    parse_judge_output: ParseJudgeOutputFn,
    resolve_rubric_criterion_score: ResolveRubricCriterionScoreFn,
) -> List[RubricCriterionJudgeResult]:
    worker_count = min(criterion_workers, len(rubric_items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(
                _run_rubric_criterion,
                idx=idx,
                criterion=criterion,
                ctx=ctx,
                candidate=candidate,
                example=example,
                response_row=response_row,
                rubric_judges=rubric_judges,
                judge_rate_limiter=judge_rate_limiter,
                build_rubric_criterion_judge_messages=build_rubric_criterion_judge_messages,
                parse_judge_output=parse_judge_output,
                resolve_rubric_criterion_score=resolve_rubric_criterion_score,
            )
            for idx, criterion in enumerate(rubric_items, start=1)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    results.sort(key=lambda item: item.criterion_index)
    return results


def _materialize_rubric_rows(
    *,
    criterion_results: List[RubricCriterionJudgeResult],
    ctx: RunnerContext,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
) -> Tuple[Dict[str, float], Dict[str, str], List[Dict[str, Any]], bool, bool, str | None, List[TraceRow], List[FailureItem]]:
    criterion_scores: Dict[str, float] = {}
    criterion_rationales: Dict[str, str] = {}
    criterion_call_details: List[Dict[str, Any]] = []
    any_parse_error = False
    any_cache_hit = False
    first_request_id = None
    trace_rows: List[TraceRow] = []
    failed_items: List[FailureItem] = []

    for result in criterion_results:
        idx = result.criterion_index
        criterion_id = result.criterion_id
        judge_model = result.judge_model
        judge_req_id = result.judge_req_id
        judge_payload = result.judge_payload
        parsed = result.parsed

        criterion_scores[criterion_id] = result.criterion_score
        criterion_rationales[criterion_id] = parsed.rationale
        any_parse_error = any_parse_error or parsed.parse_error
        any_cache_hit = any_cache_hit or result.judge_cache_hit
        if first_request_id is None:
            first_request_id = judge_payload.get("request_id") or judge_req_id

        criterion_call = {
            "criterion_id": criterion_id,
            "criterion_index": idx,
            "judge": ctx.judge_descriptor(judge_model),
            "request_id": judge_payload.get("request_id") or judge_req_id,
            "cache_key": result.judge_cache_key,
            "cache_hit": result.judge_cache_hit,
            "score": result.criterion_score,
            "raw_score": parsed.score,
            "rationale": parsed.rationale,
            "parse_error": parsed.parse_error,
            "raw": parsed.raw,
            "error": result.error,
        }
        criterion_call_details.append(criterion_call)

        trace_rows.append(
            build_judge_trace(
                run_id=ctx.run_id,
                example=example,
                judge_model=judge_model,
                request_id=criterion_call["request_id"],
                cache_key=result.judge_cache_key,
                cache_hit=result.judge_cache_hit,
                judge_messages=result.judge_messages,
                judge_payload=judge_payload,
                parsed=parsed,
                to_jsonable_messages=ctx.to_jsonable_messages,
                criterion_id=criterion_id,
                criterion_index=idx,
                criterion_score=result.criterion_score,
                error=result.error,
            )
        )

        if result.error is not None:
            failed_items.append(
                _build_judge_failure_item(
                    display_index=display_index,
                    candidate=candidate,
                    example=example,
                    error=result.error,
                    criterion_id=criterion_id,
                    judge_model=judge_model,
                    request_id=criterion_call["request_id"],
                )
            )

    return (
        criterion_scores,
        criterion_rationales,
        criterion_call_details,
        any_parse_error,
        any_cache_hit,
        first_request_id,
        trace_rows,
        failed_items,
    )


def _aggregate_rubric_result(
    *,
    criterion_scores: Dict[str, float],
    criterion_rationales: Dict[str, str],
    criterion_call_details: List[Dict[str, Any]],
    any_parse_error: bool,
    example: NormalizedExample,
    pass_threshold: float,
    apply_weighted_rubric_score: ApplyWeightedRubricScoreFn,
) -> JudgeResult:
    aggregate = JudgeResult(
        score=0.0,
        passed=False,
        rationale="\n\n".join(
            f"{criterion_id}: {text}"
            for criterion_id, text in criterion_rationales.items()
            if text
        ),
        criteria=criterion_scores,
        raw={
            "mode": "per_criterion_judges",
            "assignment": "round_robin",
            "calls": criterion_call_details,
        },
        parse_error=any_parse_error,
    )
    aggregate = apply_weighted_rubric_score(
        parsed=aggregate,
        example=example,
        pass_threshold=pass_threshold,
    )
    return _enforce_fail_closed(aggregate)


def _handle_rubric_multi_judge(
    *,
    ctx: RunnerContext,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    response_row: ResponseRow,
    criterion_workers: int,
    judge_rate_limiter: WaitableRateLimiter | None,
    build_rubric_criterion_judge_messages: BuildRubricCriterionJudgeMessagesFn,
    parse_judge_output: ParseJudgeOutputFn,
    resolve_rubric_criterion_score: ResolveRubricCriterionScoreFn,
    apply_weighted_rubric_score: ApplyWeightedRubricScoreFn,
) -> JudgeItemResult:
    rubric_items = [item for item in (example.rubric or []) if isinstance(item, dict)]
    rubric_judges = ctx.config.judges

    criterion_results = _collect_rubric_criterion_results(
        rubric_items=rubric_items,
        criterion_workers=criterion_workers,
        ctx=ctx,
        candidate=candidate,
        example=example,
        response_row=response_row,
        rubric_judges=rubric_judges,
        judge_rate_limiter=judge_rate_limiter,
        build_rubric_criterion_judge_messages=build_rubric_criterion_judge_messages,
        parse_judge_output=parse_judge_output,
        resolve_rubric_criterion_score=resolve_rubric_criterion_score,
    )

    (
        criterion_scores,
        criterion_rationales,
        criterion_call_details,
        any_parse_error,
        any_cache_hit,
        first_request_id,
        trace_rows,
        failed_items,
    ) = _materialize_rubric_rows(
        criterion_results=criterion_results,
        ctx=ctx,
        display_index=display_index,
        candidate=candidate,
        example=example,
    )

    aggregate = _aggregate_rubric_result(
        criterion_scores=criterion_scores,
        criterion_rationales=criterion_rationales,
        criterion_call_details=criterion_call_details,
        any_parse_error=any_parse_error,
        example=example,
        pass_threshold=ctx.config.run.judge_pass_threshold,
        apply_weighted_rubric_score=apply_weighted_rubric_score,
    )

    used_judges = {
        (j["judge"]["provider"], j["judge"]["model"])
        for j in criterion_call_details
    }
    if len(used_judges) == 1 and criterion_call_details:
        judge_provider_value = criterion_call_details[0]["judge"]["provider"]
        judge_model_value = criterion_call_details[0]["judge"]["model"]
    else:
        judge_provider_value = "mixed"
        judge_model_value = "mixed"

    judgment_row = build_judgment_row(
        run_id=ctx.run_id,
        run_started_at_utc=ctx.run_started_at_utc,
        example=example,
        candidate_name=candidate.name,
        judge_name="rubric_multi_judge",
        judge_provider=judge_provider_value,
        judge_model=judge_model_value,
        judge_settings={
            "assignment": "round_robin",
            "judges": [ctx.judge_descriptor(j) for j in rubric_judges],
        },
        request_id=first_request_id
        or ctx.request_id(ctx.run_id, "rubric_multi_judge", f"{example.id}:{candidate.name}", "judge"),
        cache_key=None,
        cache_hit=any_cache_hit,
        result=aggregate,
    )

    return JudgeItemResult(
        judgment_row=judgment_row,
        trace_rows=trace_rows,
        failed_items=failed_items,
    )


def _evaluate_judge_item(
    *,
    ctx: RunnerContext,
    generation_result: GenerationItemResult,
    criterion_workers: int,
    judge_rate_limiter: WaitableRateLimiter | None,
    judging_services: JudgingServices,
) -> JudgeItemResult:
    display_index = generation_result.display_index
    candidate = generation_result.candidate
    example = generation_result.example
    response_row = generation_result.response_row

    try:
        strategy = _dispatch_judge_strategy(example)
        if strategy == "mcq":
            return _handle_mcq_judgment(
                ctx=ctx,
                display_index=display_index,
                candidate=candidate,
                example=example,
                response_row=response_row,
                grade_mcq_output=judging_services.grade_mcq_output,
            )
        if strategy == "rubric_multi_judge":
            return _handle_rubric_multi_judge(
                ctx=ctx,
                display_index=display_index,
                candidate=candidate,
                example=example,
                response_row=response_row,
                criterion_workers=criterion_workers,
                judge_rate_limiter=judge_rate_limiter,
                build_rubric_criterion_judge_messages=judging_services.build_rubric_criterion_judge_messages,
                parse_judge_output=judging_services.parse_judge_output,
                resolve_rubric_criterion_score=judging_services.resolve_rubric_criterion_score,
                apply_weighted_rubric_score=judging_services.apply_weighted_rubric_score,
            )
        return _handle_single_judge(
            ctx=ctx,
            display_index=display_index,
            candidate=candidate,
            example=example,
            response_row=response_row,
            judge_rate_limiter=judge_rate_limiter,
            build_judge_messages=judging_services.build_judge_messages,
            parse_judge_output=judging_services.parse_judge_output,
            apply_weighted_rubric_score=judging_services.apply_weighted_rubric_score,
            apply_policy_score_postprocessing=judging_services.apply_policy_score_postprocessing,
        )
    except Exception as exc:  # noqa: BLE001 - explicit outer phase boundary
        return _build_unexpected_judge_item_result(
            ctx=ctx,
            display_index=display_index,
            candidate=candidate,
            example=example,
            error=f"{type(exc).__name__}: {exc}",
        )


def _handle_single_judge(
    *,
    ctx: RunnerContext,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    response_row: ResponseRow,
    judge_rate_limiter: WaitableRateLimiter | None,
    build_judge_messages: BuildJudgeMessagesFn,
    parse_judge_output: ParseJudgeOutputFn,
    apply_weighted_rubric_score: ApplyWeightedRubricScoreFn,
    apply_policy_score_postprocessing: ApplyPolicyScorePostprocessingFn,
) -> JudgeItemResult:
    judge_model = ctx.config.primary_judge
    judge_provider = ctx.providers[judge_model.provider]
    judge_messages = build_judge_messages(
        example,
        response_row["response_text"],
        pass_threshold=ctx.config.run.judge_pass_threshold,
    )
    judge_req_id = ctx.request_id(
        ctx.run_id,
        judge_model.name,
        f"{example.id}:{candidate.name}",
        stage="judge",
    )

    judge_call = _run_judge_call(
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_req_id=judge_req_id,
        judge_messages=judge_messages,
        build_request=ctx.build_request,
        run_model_call=ctx.run_model_call,
        parse_judge_output=parse_judge_output,
        cache=ctx.cache,
        config=ctx.config,
        judge_rate_limiter=judge_rate_limiter,
        google_provider_names=ctx.google_provider_names,
        error_result_builder=build_error_judge_result,
    )

    parsed = apply_weighted_rubric_score(
        parsed=judge_call.parsed,
        example=example,
        pass_threshold=ctx.config.run.judge_pass_threshold,
    )
    parsed = apply_policy_score_postprocessing(
        parsed=parsed,
        example=example,
        pass_threshold=ctx.config.run.judge_pass_threshold,
    )
    parsed = _enforce_fail_closed(parsed)

    judgment_row = build_judgment_row(
        run_id=ctx.run_id,
        run_started_at_utc=ctx.run_started_at_utc,
        example=example,
        candidate_name=candidate.name,
        judge_name=judge_model.name,
        judge_provider=judge_model.provider,
        judge_model=judge_model.model,
        judge_settings=ctx.model_settings(judge_model),
        request_id=judge_call.judge_payload.get("request_id") or judge_req_id,
        cache_key=judge_call.judge_cache_key,
        cache_hit=judge_call.judge_cache_hit,
        result=parsed,
    )

    trace_row = build_judge_trace(
        run_id=ctx.run_id,
        example=example,
        judge_model=judge_model,
        request_id=judgment_row["request_id"],
        cache_key=judge_call.judge_cache_key,
        cache_hit=judge_call.judge_cache_hit,
        judge_messages=judge_messages,
        judge_payload=judge_call.judge_payload,
        parsed=parsed,
        to_jsonable_messages=ctx.to_jsonable_messages,
    )

    failed_items: List[FailureItem] = []
    if judge_call.error is not None:
        failed_items.append(
            _build_judge_failure_item(
                display_index=display_index,
                candidate=candidate,
                example=example,
                error=judge_call.error,
                judge_model=judge_model,
                request_id=judgment_row["request_id"],
            )
        )

    return JudgeItemResult(
        judgment_row=judgment_row,
        trace_rows=[trace_row],
        failed_items=failed_items,
    )


def _build_unexpected_judge_item_result(
    *,
    ctx: RunnerContext,
    display_index: int,
    candidate: ModelConfig,
    example: NormalizedExample,
    error: str,
) -> JudgeItemResult:
    parsed = _enforce_fail_closed(build_error_judge_result(error, context="judge dispatch"))
    judge_model = ctx.config.primary_judge
    request_id = ctx.request_id(
        ctx.run_id,
        "judge_phase_error",
        f"{example.id}:{candidate.name}",
        stage="judge",
    )
    payload: ModelCallPayload = {
        "text": "",
        "usage": {},
        "latency_s": None,
        "request_id": request_id,
        "error": error,
    }

    judgment_row = build_judgment_row(
        run_id=ctx.run_id,
        run_started_at_utc=ctx.run_started_at_utc,
        example=example,
        candidate_name=candidate.name,
        judge_name="judge_phase_error",
        judge_provider="internal",
        judge_model="internal",
        judge_settings={},
        request_id=request_id,
        cache_key=None,
        cache_hit=False,
        result=parsed,
    )
    trace_row = build_judge_trace(
        run_id=ctx.run_id,
        example=example,
        judge_model=judge_model,
        request_id=request_id,
        cache_key=None,
        cache_hit=False,
        judge_messages=[],
        judge_payload=payload,
        parsed=parsed,
        to_jsonable_messages=ctx.to_jsonable_messages,
        error=error,
    )

    return JudgeItemResult(
        judgment_row=judgment_row,
        trace_rows=[trace_row],
        failed_items=[
            _build_judge_failure_item(
                display_index=display_index,
                candidate=candidate,
                example=example,
                error=error,
                judge_model=judge_model,
                request_id=request_id,
            )
        ],
    )


def _dispatch_judge_strategy(example: NormalizedExample) -> str:
    if example.judge_mode == "mcq":
        return "mcq"
    if example.judge_mode == "rubric" and isinstance(example.rubric, list) and example.rubric:
        return "rubric_multi_judge"
    return "single_judge"


def run_judge_phase(
    *,
    ctx: RunnerContext,
    total_items: int,
    generation_results: List[GenerationItemResult],
    judge_workers: int,
    judge_rate_limiter: WaitableRateLimiter | None,
    judging_services: JudgingServices,
) -> JudgingPhaseResult:
    ctx.emit_progress(ctx.progress_mode, ctx.progress_line(stage="judge_phase_start", total_items=total_items))

    judgments_rows = []
    trace_rows = []
    failed_items: List[FailureItem] = []
    interrupted = False
    item_workers = min(max(1, judge_workers), len(generation_results) or 1)
    criterion_workers = max(1, judge_workers // item_workers)
    ordered_results: Dict[int, JudgeItemResult] = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=item_workers)
    futures: Dict[concurrent.futures.Future[JudgeItemResult], int] = {}

    try:
        futures = {
            pool.submit(
                _evaluate_judge_item,
                ctx=ctx,
                generation_result=generation_result,
                criterion_workers=criterion_workers,
                judge_rate_limiter=judge_rate_limiter,
                judging_services=judging_services,
            ): generation_result.display_index
            for generation_result in generation_results
        }
        for future in concurrent.futures.as_completed(futures):
            display_index = futures[future]
            ordered_results[display_index] = future.result()
    except BaseException as exc:
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        interrupted = True
        for future in futures:
            future.cancel()
        for future, display_index in futures.items():
            if display_index in ordered_results or future.cancelled() or not future.done():
                continue
            try:
                ordered_results[display_index] = future.result()
            except BaseException:
                continue
        pool.shutdown(wait=False, cancel_futures=True)
        ctx.emit_progress(
            ctx.progress_mode,
            ctx.progress_line(
                stage="judge_phase_interrupted",
                total_items=total_items,
                completed_items=len(ordered_results),
            ),
        )
    else:
        pool.shutdown(wait=True, cancel_futures=False)

    for generation_result in generation_results:
        display_index = generation_result.display_index
        candidate = generation_result.candidate
        example = generation_result.example
        if display_index not in ordered_results:
            continue
        result = ordered_results[display_index]

        judgment_row = result.judgment_row
        judgments_rows.append(judgment_row)
        trace_rows.extend(result.trace_rows)
        failed_items.extend(result.failed_items)
        ctx.emit_progress(
            ctx.progress_mode,
            ctx.progress_line(
                item=f"{display_index}/{total_items}",
                stage="judge_done",
                candidate=candidate.name,
                dataset=example.dataset_name,
                example=example.id,
                judge_mode=example.judge_mode,
                score=judgment_row["score"],
                passed=judgment_row["pass"],
                parse_error=judgment_row["parse_error"],
            ),
        )

    if not interrupted:
        ctx.emit_progress(ctx.progress_mode, ctx.progress_line(stage="judge_phase_done", total_items=total_items))

    return JudgingPhaseResult(
        judgments_rows=judgments_rows,
        trace_rows=trace_rows,
        failed_items=failed_items,
        interrupted=interrupted,
        interrupted_stage="judging" if interrupted else None,
    )
