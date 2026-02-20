from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple

from src.cache import DiskCache
from src.config import BenchmarkConfig, ModelConfig
from src.config import RetryConfig
from src.providers.base import BaseProvider
from src.runner.contracts import ModelCallPayload
from src.runner.rate_limiter import PerMinuteRateLimiter
from src.types import JudgeResult, LLMMessage, LLMRequest, NormalizedExample


ValidateCanonicalInputsFn = Callable[[BenchmarkConfig], None]
LoadAllExamplesFn = Callable[[BenchmarkConfig], tuple[List[NormalizedExample], List[Dict[str, Any]]]]
RequiredProviderNamesFn = Callable[[BenchmarkConfig], Set[str]]
BuildProviderFn = Callable[[str, BenchmarkConfig], BaseProvider]
RunModelCallFn = Callable[
    [
        BaseProvider,
        LLMRequest,
        DiskCache,
        RetryConfig,
        str,
        bool,
        Callable[[int], None] | None,
    ],
    Tuple[ModelCallPayload, bool, str | None],
]

BuildCandidateMessagesFn = Callable[[NormalizedExample, str], List[LLMMessage]]
GradeMcqOutputFn = Callable[[NormalizedExample, str, float], JudgeResult]
BuildJudgeMessagesFn = Callable[[NormalizedExample, str, float], List[LLMMessage]]
BuildRubricCriterionJudgeMessagesFn = Callable[
    [NormalizedExample, str, Dict[str, Any], int, float],
    List[LLMMessage],
]
ParseJudgeOutputFn = Callable[[str, float], JudgeResult]
ResolveRubricCriterionScoreFn = Callable[[Dict[str, float], Dict[str, Any], int, float], tuple[float, bool]]
ApplyWeightedRubricScoreFn = Callable[[JudgeResult, NormalizedExample, float], JudgeResult]
ApplyPolicyScorePostprocessingFn = Callable[[JudgeResult, NormalizedExample, float], JudgeResult]

EmitProgressFn = Callable[[str, str], None]
ProgressLineFn = Callable[..., str]
RequestIdFn = Callable[[str, str, str, str], str]
ToJsonableMessagesFn = Callable[[List[LLMMessage]], List[Dict[str, str]]]
BuildRequestFn = Callable[[ModelConfig, List[LLMMessage], str], LLMRequest]
ModelSettingsFn = Callable[[ModelConfig], Dict[str, Any]]
JudgeDescriptorFn = Callable[[ModelConfig], Dict[str, Any]]


@dataclass(frozen=True)
class BootstrapServices:
    validate_canonical_inputs: ValidateCanonicalInputsFn
    load_all_examples: LoadAllExamplesFn
    required_provider_names: RequiredProviderNamesFn
    build_provider: BuildProviderFn


@dataclass(frozen=True)
class GenerationServices:
    run_model_call: RunModelCallFn
    build_candidate_messages: BuildCandidateMessagesFn


@dataclass(frozen=True)
class JudgingServices:
    grade_mcq_output: GradeMcqOutputFn
    build_judge_messages: BuildJudgeMessagesFn
    build_rubric_criterion_judge_messages: BuildRubricCriterionJudgeMessagesFn
    parse_judge_output: ParseJudgeOutputFn
    resolve_rubric_criterion_score: ResolveRubricCriterionScoreFn
    apply_weighted_rubric_score: ApplyWeightedRubricScoreFn
    apply_policy_score_postprocessing: ApplyPolicyScorePostprocessingFn


@dataclass(frozen=True)
class InfrastructureServices:
    per_minute_rate_limiter_cls: type[PerMinuteRateLimiter]
    emit_progress: EmitProgressFn
    progress_line: ProgressLineFn
    request_id: RequestIdFn
    to_jsonable_messages: ToJsonableMessagesFn
    build_request: BuildRequestFn
    model_settings: ModelSettingsFn
    judge_descriptor: JudgeDescriptorFn


@dataclass(frozen=True)
class RunnerServices:
    bootstrap: BootstrapServices
    generation: GenerationServices
    judging: JudgingServices
    infrastructure: InfrastructureServices
