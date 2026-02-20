from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from .types import (
    FINAL_RESPONSE_SOURCES,
    RESPONSE_APIS,
    FinalResponseSource,
    JudgeMode,
    ResponseAPI,
)

MAX_RESPONSE_RPM = 50


@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


@dataclass
class CacheConfig:
    enabled: bool = True
    dir: str = "cache"


@dataclass
class DatasetConfig:
    name: str
    path: str
    provenance: str = "auto"
    judge_mode: JudgeMode = "auto"
    enabled: bool = True
    split_field: Optional[str] = None
    split_value: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class DataConfig:
    datasets: List[DatasetConfig] = field(default_factory=list)


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    runs_root: str = "data/runs"
    output_dir: str = "outputs"
    default_system_prompt: str = (
        "You are a careful legal reasoning assistant. Answer clearly and concisely, "
        "state uncertainty when needed, and avoid fabrication."
    )
    final_response_source: FinalResponseSource = "sampled"
    prefilled_responses_path: Optional[str] = None
    previous_output_path: Optional[str] = None
    response_api: ResponseAPI = "chat.completions"
    use_scratchpad: bool = False
    web_search: bool = False
    judge_pass_threshold: float = 0.7
    response_parallel_workers: int = 8
    response_rate_limit_rpm: int = 50
    provider_response_rate_limit_rpm: Dict[str, int] = field(default_factory=dict)
    judge_parallel_workers: int = 4
    judge_rate_limit_rpm: int = 12
    include_raw_provider_response: bool = False


@dataclass
class ProviderConfig:
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: int = 120
    project: Optional[str] = None
    location: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    temperature: float = 0.0
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    extra_body: Optional[Dict[str, object]] = None

    def to_settings_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "thinking_budget": self.thinking_budget,
            "extra_body": self.extra_body,
        }


@dataclass
class BenchmarkConfig:
    providers: Dict[str, ProviderConfig]
    candidates: List[ModelConfig]
    judges: List[ModelConfig]
    data: DataConfig = field(default_factory=DataConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        providers_raw = raw.get("providers", {})
        providers = {k: ProviderConfig(**(v or {})) for k, v in providers_raw.items()}

        candidates = [ModelConfig(**m) for m in raw.get("candidates", [])]
        judges = [ModelConfig(**m) for m in raw.get("judges", [])]
        if not judges:
            raise ValueError("Config must define at least one judge model in 'judges'.")

        datasets = [DatasetConfig(**d) for d in raw.get("data", {}).get("datasets", [])]
        data = DataConfig(datasets=datasets)

        retry = RetryConfig(**raw.get("retry", {}))
        cache = CacheConfig(**raw.get("cache", {}))
        run = RunConfig(**raw.get("run", {}))

        cfg = cls(
            providers=providers,
            candidates=candidates,
            judges=judges,
            data=data,
            retry=retry,
            cache=cache,
            run=run,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.candidates:
            raise ValueError("Config must define at least one candidate model in 'candidates'.")

        if not self.data.datasets:
            raise ValueError("Config must define at least one dataset in data.datasets.")

        if not self.judges:
            raise ValueError("Config must define at least one judge model in 'judges'.")

        known = set(self.providers.keys())
        for m in [*self.candidates, *self.judges]:
            if m.provider not in known:
                raise ValueError(f"Model '{m.name}' references unknown provider '{m.provider}'.")

        enabled_count = 0
        for ds in self.data.datasets:
            if not ds.enabled:
                continue
            enabled_count += 1
            if not Path(ds.path).exists():
                raise ValueError(f"Dataset path does not exist: {ds.path}")

        if enabled_count == 0:
            raise ValueError("At least one dataset must have enabled=true.")

        rpm = int(self.run.response_rate_limit_rpm)
        if rpm <= 0:
            raise ValueError("run.response_rate_limit_rpm must be a positive integer.")
        if rpm > MAX_RESPONSE_RPM:
            raise ValueError(f"run.response_rate_limit_rpm must be <= {MAX_RESPONSE_RPM}.")

        for provider_name, provider_rpm in self.run.provider_response_rate_limit_rpm.items():
            if provider_name not in known:
                raise ValueError(
                    f"run.provider_response_rate_limit_rpm references unknown provider '{provider_name}'."
                )
            provider_rpm_int = int(provider_rpm)
            if provider_rpm_int <= 0:
                raise ValueError(
                    f"run.provider_response_rate_limit_rpm['{provider_name}'] must be a positive integer."
                )
            if provider_rpm_int > MAX_RESPONSE_RPM:
                raise ValueError(
                    f"run.provider_response_rate_limit_rpm['{provider_name}'] must be <= {MAX_RESPONSE_RPM}."
                )

        if self.run.judge_parallel_workers <= 0:
            raise ValueError("run.judge_parallel_workers must be a positive integer.")

        if self.run.judge_rate_limit_rpm < 0:
            raise ValueError("run.judge_rate_limit_rpm must be >= 0.")

        if self.run.final_response_source not in FINAL_RESPONSE_SOURCES:
            raise ValueError(
                "run.final_response_source must be one of: sampled, prefilled, part_of_conversation."
            )

        if self.run.final_response_source == "prefilled":
            if not self.run.prefilled_responses_path:
                raise ValueError(
                    "run.prefilled_responses_path must be set when run.final_response_source=prefilled."
                )
            if not Path(self.run.prefilled_responses_path).exists():
                raise ValueError(
                    f"run.prefilled_responses_path does not exist: {self.run.prefilled_responses_path}"
                )
        if self.run.final_response_source == "part_of_conversation":
            if not self.run.previous_output_path:
                raise ValueError(
                    "run.previous_output_path must be set when "
                    "run.final_response_source=part_of_conversation."
                )
            if not Path(self.run.previous_output_path).exists():
                raise ValueError(
                    f"run.previous_output_path does not exist: {self.run.previous_output_path}"
                )

        if self.run.response_api not in RESPONSE_APIS:
            raise ValueError("run.response_api must be one of: responses, chat.completions.")
        if self.run.final_response_source == "sampled":
            from .providers.base import provider_supported_response_apis

            for candidate in self.candidates:
                supported = provider_supported_response_apis(candidate.provider)
                if self.run.response_api not in supported:
                    supported_text = ", ".join(sorted(supported))
                    raise ValueError(
                        f"Model '{candidate.name}' provider '{candidate.provider}' does not support "
                        f"run.response_api='{self.run.response_api}'. Supported: {supported_text}."
                    )

        for m in [*self.candidates, *self.judges]:
            if m.max_tokens is not None and m.max_tokens <= 0:
                raise ValueError(
                    f"Model '{m.name}' has max_tokens={m.max_tokens}; must be > 0 when set."
                )
            if m.thinking_budget is not None and m.thinking_budget < 0:
                raise ValueError(
                    f"Model '{m.name}' has thinking_budget={m.thinking_budget}; must be >= 0 when set."
                )

        candidate_names = [m.name for m in self.candidates]
        if len(candidate_names) != len(set(candidate_names)):
            dupes = sorted(
                n for n in set(candidate_names) if candidate_names.count(n) > 1
            )
            raise ValueError(f"Duplicate candidate name(s): {', '.join(dupes)}")

        judge_names = [m.name for m in self.judges]
        if len(judge_names) != len(set(judge_names)):
            dupes = sorted(
                n for n in set(judge_names) if judge_names.count(n) > 1
            )
            raise ValueError(f"Duplicate judge name(s): {', '.join(dupes)}")

    def provider_env(self, provider_name: str) -> Optional[str]:
        provider = self.providers.get(provider_name)
        if not provider or not provider.api_key_env:
            return None
        return os.getenv(provider.api_key_env)

    @property
    def primary_judge(self) -> ModelConfig:
        if not self.judges:
            raise ValueError("Config must define at least one judge model in 'judges'.")
        return self.judges[0]
