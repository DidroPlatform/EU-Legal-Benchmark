from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, FrozenSet

from ..config import ProviderConfig
from ..types import GOOGLE_PROVIDER_NAMES, LLMRequest, LLMResponse


class BaseProvider(ABC):
    def __init__(self, provider_name: str, config: ProviderConfig, api_key: str | None):
        self.provider_name = provider_name
        self.config = config
        self.api_key = api_key

    @abstractmethod
    def generate(self, request: LLMRequest, include_raw: bool = False) -> LLMResponse:
        raise NotImplementedError

    @classmethod
    def supported_response_apis(cls) -> FrozenSet[str]:
        return frozenset({"chat.completions"})

    def close(self) -> None:
        return None


def usage_dict(prompt_tokens: int | None, completion_tokens: int | None) -> Dict[str, int | None]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None or completion_tokens is not None
            else None
        ),
    }


def provider_supported_response_apis(provider_name: str) -> FrozenSet[str]:
    if provider_name in GOOGLE_PROVIDER_NAMES:
        return frozenset({"chat.completions"})
    # Non-Google providers route through LiteLLM adapter in this repository.
    return frozenset({"chat.completions", "responses"})
