from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet

from src.cache import DiskCache
from src.config import BenchmarkConfig
from src.providers.base import BaseProvider
from src.runner.services import (
    BuildRequestFn,
    EmitProgressFn,
    JudgeDescriptorFn,
    ModelSettingsFn,
    ProgressLineFn,
    RequestIdFn,
    RunModelCallFn,
    ToJsonableMessagesFn,
)
@dataclass
class RunnerContext:
    """Shared infrastructure for a single benchmark run.

    Bundles the configuration, runtime state, and utility callbacks that
    both the generation and judging phases need, replacing 15+ repeated
    keyword arguments across the two phase functions and their helpers.
    """

    # Run identity
    config: BenchmarkConfig
    run_id: str
    run_started_at_utc: str

    # Infrastructure
    providers: Dict[str, BaseProvider]
    cache: DiskCache
    progress_mode: str
    google_provider_names: FrozenSet[str]

    # Utility callbacks
    emit_progress: EmitProgressFn
    progress_line: ProgressLineFn
    model_settings: ModelSettingsFn
    to_jsonable_messages: ToJsonableMessagesFn
    request_id: RequestIdFn
    build_request: BuildRequestFn
    run_model_call: RunModelCallFn
    judge_descriptor: JudgeDescriptorFn
