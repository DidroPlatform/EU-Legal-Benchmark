from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from src.cache import DiskCache
from src.config import BenchmarkConfig, RetryConfig
from src.data.loader import load_examples
from src.data.schema import validate_jsonl_file
from src.judge.judge import (
    apply_policy_score_postprocessing,
    apply_weighted_rubric_score,
    build_judge_messages,
    build_rubric_criterion_judge_messages,
    parse_judge_output,
    resolve_rubric_criterion_score,
)
from src.judge.mcq import grade_mcq_output
from src.prompting.templates import build_candidate_messages
from src.providers import build_provider
from src.providers.base import BaseProvider
from src.retry import with_retries
from src.runner.helpers import (
    _build_request,
    _cache_key_payload,
    _emit_progress,
    _judge_descriptor,
    _model_settings,
    _progress_line,
    _request_id,
    _to_jsonable_messages,
)
from src.runner.rate_limiter import PerMinuteRateLimiter
from src.runner.services import (
    ApplyPolicyScorePostprocessingFn,
    ApplyWeightedRubricScoreFn,
    BootstrapServices,
    BuildCandidateMessagesFn,
    BuildJudgeMessagesFn,
    BuildProviderFn,
    BuildRubricCriterionJudgeMessagesFn,
    GradeMcqOutputFn,
    GenerationServices,
    InfrastructureServices,
    JudgingServices,
    LoadAllExamplesFn,
    ParseJudgeOutputFn,
    RequiredProviderNamesFn,
    ResolveRubricCriterionScoreFn,
    RunModelCallFn,
    ValidateCanonicalInputsFn,
    RunnerServices,
)
from src.setup_checks import required_provider_names
from src.types import LLMRequest, LLMResponse, NormalizedExample


def run_model_call(
    provider: BaseProvider,
    request: LLMRequest,
    cache: DiskCache,
    retry_cfg: RetryConfig,
    stage: str,
    include_raw: bool,
    before_attempt: Callable[[int], None] | None = None,
) -> Tuple[Dict[str, Any], bool, str]:
    def _is_empty_response_payload(payload: Dict[str, Any]) -> bool:
        text = payload.get("text")
        if text is None:
            return True
        return not str(text).strip()

    key = cache.make_key(_cache_key_payload(request, stage=stage))
    cached = cache.get(key)
    if cached is not None:
        if stage == "response" and _is_empty_response_payload(cached):
            cache.delete(key)
        else:
            return cached, True, key

    def _generate() -> LLMResponse:
        response = provider.generate(request, include_raw=include_raw)
        if stage == "response":
            text = response.text
            if text is None or not str(text).strip():
                raise RuntimeError("empty response text")
        return response

    response = with_retries(
        _generate,
        retry_cfg,
        before_attempt=before_attempt,
    )

    payload = {
        "provider": response.provider,
        "model": response.model,
        "text": response.text,
        "usage": response.usage,
        "latency_s": response.latency_s,
        "request_id": response.request_id,
        "raw_response": response.raw_response,
    }
    if stage == "response" and _is_empty_response_payload(payload):
        raise RuntimeError("empty response text")
    cache.set(key, payload)
    return payload, False, key


def load_all_examples(config: BenchmarkConfig) -> Tuple[List[NormalizedExample], List[Dict[str, Any]]]:
    examples: List[NormalizedExample] = []
    dataset_stats: List[Dict[str, Any]] = []

    for dataset in config.data.datasets:
        if not dataset.enabled:
            continue
        rows = load_examples(dataset)
        examples.extend(rows)
        dataset_stats.append(
            {
                "dataset": dataset.name,
                "path": dataset.path,
                "provenance": dataset.provenance,
                "judge_mode": dataset.judge_mode,
                "selected_examples": len(rows),
            }
        )

    return examples, dataset_stats


def validate_canonical_inputs(config: BenchmarkConfig) -> None:
    problems: List[str] = []

    for dataset in config.data.datasets:
        if not dataset.enabled:
            continue
        result = validate_jsonl_file(dataset.path)
        invalid = int(result.get("invalid_rows", 0))
        if invalid <= 0:
            continue

        rows = int(result.get("rows", 0))
        details = []
        for err in result.get("errors", [])[:5]:
            line = err.get("line")
            row_id = err.get("id")
            msg = "; ".join(err.get("errors", []))
            details.append(f"line={line}, id={row_id}: {msg}")
        detail_text = "\n      ".join(details) if details else "(no details)"

        problems.append(
            f"- dataset='{dataset.name}' path='{dataset.path}' invalid_rows={invalid}/{rows}\n"
            f"      {detail_text}"
        )

    if problems:
        raise ValueError(
            "Canonical input validation failed for legal_eval_v1.\n"
            "Fix the dataset files before running the benchmark.\n"
            + "\n".join(problems)
        )


def build_runner_services(
    *,
    validate_canonical_inputs_fn: ValidateCanonicalInputsFn = validate_canonical_inputs,
    load_all_examples_fn: LoadAllExamplesFn = load_all_examples,
    required_provider_names_fn: RequiredProviderNamesFn = required_provider_names,
    build_provider_fn: BuildProviderFn = build_provider,
    run_model_call_fn: RunModelCallFn = run_model_call,
    build_candidate_messages_fn: BuildCandidateMessagesFn = build_candidate_messages,
    grade_mcq_output_fn: GradeMcqOutputFn = grade_mcq_output,
    build_judge_messages_fn: BuildJudgeMessagesFn = build_judge_messages,
    build_rubric_criterion_judge_messages_fn: BuildRubricCriterionJudgeMessagesFn = (
        build_rubric_criterion_judge_messages
    ),
    parse_judge_output_fn: ParseJudgeOutputFn = parse_judge_output,
    resolve_rubric_criterion_score_fn: ResolveRubricCriterionScoreFn = resolve_rubric_criterion_score,
    apply_weighted_rubric_score_fn: ApplyWeightedRubricScoreFn = apply_weighted_rubric_score,
    apply_policy_score_postprocessing_fn: ApplyPolicyScorePostprocessingFn = (
        apply_policy_score_postprocessing
    ),
) -> RunnerServices:
    return RunnerServices(
        bootstrap=BootstrapServices(
            validate_canonical_inputs=validate_canonical_inputs_fn,
            load_all_examples=load_all_examples_fn,
            required_provider_names=required_provider_names_fn,
            build_provider=build_provider_fn,
        ),
        generation=GenerationServices(
            run_model_call=run_model_call_fn,
            build_candidate_messages=build_candidate_messages_fn,
        ),
        judging=JudgingServices(
            grade_mcq_output=grade_mcq_output_fn,
            build_judge_messages=build_judge_messages_fn,
            build_rubric_criterion_judge_messages=build_rubric_criterion_judge_messages_fn,
            parse_judge_output=parse_judge_output_fn,
            resolve_rubric_criterion_score=resolve_rubric_criterion_score_fn,
            apply_weighted_rubric_score=apply_weighted_rubric_score_fn,
            apply_policy_score_postprocessing=apply_policy_score_postprocessing_fn,
        ),
        infrastructure=InfrastructureServices(
            per_minute_rate_limiter_cls=PerMinuteRateLimiter,
            emit_progress=_emit_progress,
            progress_line=_progress_line,
            request_id=_request_id,
            to_jsonable_messages=_to_jsonable_messages,
            build_request=_build_request,
            model_settings=_model_settings,
            judge_descriptor=_judge_descriptor,
        ),
    )
