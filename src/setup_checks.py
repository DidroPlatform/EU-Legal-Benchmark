from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import List, Set

from .config import BenchmarkConfig
from .providers.base import GOOGLE_PROVIDER_NAMES, provider_supported_response_apis


_PROVIDER_IMPORTS = {
    "litellm": "litellm",
    "google-genai": "google.genai",
}


_VERTEX_AUTH_IMPORT = "google.auth"
_BEDROCK_IMPORT = "boto3"


@dataclass
class SetupReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def required_provider_names(config: BenchmarkConfig) -> Set[str]:
    names = {judge.provider for judge in config.judges}
    if config.run.final_response_source == "sampled":
        names.update(candidate.provider for candidate in config.candidates)
    return names


def check_setup(config: BenchmarkConfig) -> SetupReport:
    report = SetupReport()
    providers = required_provider_names(config)
    uses_google_genai = any(name in GOOGLE_PROVIDER_NAMES for name in providers)
    uses_litellm = any(name not in GOOGLE_PROVIDER_NAMES for name in providers)

    if uses_litellm:
        litellm_module = _PROVIDER_IMPORTS["litellm"]
        if importlib.util.find_spec(litellm_module) is None:
            report.errors.append(f"Provider setup requires Python package '{litellm_module}' but it is not installed.")
            return report

    if uses_google_genai:
        genai_module = _PROVIDER_IMPORTS["google-genai"]
        if importlib.util.find_spec(genai_module) is None:
            report.errors.append(f"Provider setup requires Python package '{genai_module}' but it is not installed.")
            return report

    configured_candidates = (
        list(config.candidates) if config.run.final_response_source == "sampled" else []
    )
    configured_models = [*configured_candidates, *config.judges]
    uses_bedrock_models = any(model.model.strip().startswith("bedrock/") for model in configured_models)
    if uses_bedrock_models and importlib.util.find_spec(_BEDROCK_IMPORT) is None:
        report.errors.append(
            "Bedrock model routing requires Python package 'boto3' but it is not installed. "
            "Install with: pip install boto3."
        )
        return report

    for name in sorted(providers):
        pconf = config.providers.get(name)
        if pconf is None:
            report.errors.append(f"Provider '{name}' is referenced by a model but is missing from providers config.")
            continue

        if pconf.api_key_env:
            value = os.getenv(pconf.api_key_env)
            if not value:
                report.errors.append(
                    f"Provider '{name}' requires env var '{pconf.api_key_env}' but it is not set."
                )
        elif name in {"openai", "mistral", "litellm"} or name in GOOGLE_PROVIDER_NAMES:
            report.warnings.append(
                f"Provider '{name}' has no api_key_env configured; relying on provider default environment detection."
            )

        if name in {"vertex", "anthropic_vertex"}:
            vertex_project = pconf.project or os.getenv("VERTEXAI_PROJECT")
            vertex_location = pconf.location or os.getenv("VERTEXAI_LOCATION")
            if not vertex_project or not vertex_location:
                report.errors.append(
                    f"Provider '{name}' requires Vertex project+location via config "
                    "or VERTEXAI_PROJECT / VERTEXAI_LOCATION env vars."
                )
            if not pconf.api_key_env and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                report.warnings.append(
                    f"Provider '{name}' usually needs ADC when no API key is configured; "
                    "GOOGLE_APPLICATION_CREDENTIALS is not set."
                )
            if importlib.util.find_spec(_VERTEX_AUTH_IMPORT) is None:
                report.errors.append(
                    f"Provider '{name}' requires Google auth dependencies for Vertex. "
                    "Install with: pip install 'litellm[google]' "
                    "or pip install google-cloud-aiplatform."
                )

    if config.run.final_response_source == "sampled":
        for candidate in config.candidates:
            supported = provider_supported_response_apis(candidate.provider)
            if config.run.response_api not in supported:
                report.errors.append(
                    f"Model '{candidate.name}' provider '{candidate.provider}' does not support "
                    f"run.response_api='{config.run.response_api}'. "
                    f"Supported: {', '.join(sorted(supported))}."
                )
        if uses_litellm and config.run.response_api == "responses":
            try:
                import litellm  # type: ignore
            except Exception:
                litellm = None  # pragma: no cover
            if litellm is None or not callable(getattr(litellm, "responses", None)):
                report.errors.append(
                    "run.response_api='responses' requires LiteLLM responses API support, "
                    "but current litellm package does not expose callable `responses`."
                )

    return report
