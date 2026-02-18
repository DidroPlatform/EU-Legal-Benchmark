from __future__ import annotations

from ..config import BenchmarkConfig
from .base import BaseProvider, GOOGLE_PROVIDER_NAMES
from .google_genai import GoogleGenAIProvider
from .litellm import LiteLLMProvider


def build_provider(provider_name: str, config: BenchmarkConfig) -> BaseProvider:
    pconf = config.providers[provider_name]
    api_key = config.provider_env(provider_name)
    if provider_name in GOOGLE_PROVIDER_NAMES:
        return GoogleGenAIProvider(provider_name, pconf, api_key)
    return LiteLLMProvider(provider_name, pconf, api_key)
