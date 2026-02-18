from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..config import ProviderConfig
from ..types import LLMRequest, LLMResponse

GOOGLE_PROVIDER_NAMES: frozenset[str] = frozenset({"google_genai", "google-genai", "gemini"})


class BaseProvider(ABC):
    def __init__(self, provider_name: str, config: ProviderConfig, api_key: str | None):
        self.provider_name = provider_name
        self.config = config
        self.api_key = api_key

    @abstractmethod
    def generate(self, request: LLMRequest, include_raw: bool = False) -> LLMResponse:
        raise NotImplementedError

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
