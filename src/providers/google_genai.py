from __future__ import annotations

import time
from typing import Any, Dict, List

from .base import BaseProvider, usage_dict
from ..config import ProviderConfig
from ..types import LLMMessage, LLMRequest, LLMResponse


class GoogleGenAIProvider(BaseProvider):
    def __init__(self, provider_name: str, config: ProviderConfig, api_key: str | None):
        super().__init__(provider_name, config, api_key)
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install dependency 'google-genai' to use Google GenAI provider.") from exc

        self._genai = genai
        self._types = genai_types
        # google-genai expects timeout in milliseconds; config timeout is in seconds.
        # Also clamp to API minimum deadline of 10 seconds to avoid INVALID_ARGUMENT.
        timeout_ms = max(int(config.timeout_s), 10) * 1000
        http_options = self._types.HttpOptions(timeout=timeout_ms)
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    def close(self) -> None:
        self._client.close()

    @classmethod
    def supported_response_apis(cls) -> frozenset[str]:
        return frozenset({"chat.completions"})

    def _split_messages(self, messages: List[LLMMessage]) -> tuple[List[LLMMessage], List[LLMMessage]]:
        system_msgs = [m for m in messages if m.role == "system"]
        non_system_msgs = [m for m in messages if m.role != "system"]
        return system_msgs, non_system_msgs

    def _to_contents(self, messages: List[LLMMessage]) -> List[Any]:
        role_map = {
            "assistant": "model",
            "user": "user",
            "model": "model",
        }
        contents = []
        for msg in messages:
            role = role_map.get(msg.role, "user")
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part.from_text(text=msg.content)],
                )
            )
        return contents

    def generate(self, request: LLMRequest, include_raw: bool = False) -> LLMResponse:
        if request.response_api != "chat.completions":
            raise ValueError(
                f"Provider '{self.provider_name}' does not support request.response_api='{request.response_api}'. "
                "Supported: chat.completions."
            )
        start = time.perf_counter()
        system_msgs, non_system_msgs = self._split_messages(request.messages)
        config_kwargs: Dict[str, Any] = {}

        if system_msgs:
            config_kwargs["system_instruction"] = "\n\n".join(m.content for m in system_msgs)
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            config_kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            config_kwargs["max_output_tokens"] = request.max_tokens
        if request.frequency_penalty is not None:
            config_kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            config_kwargs["presence_penalty"] = request.presence_penalty
        if request.seed is not None:
            config_kwargs["seed"] = request.seed
        if request.reasoning_effort is not None:
            config_kwargs["reasoning_effort"] = request.reasoning_effort
        if request.thinking_budget is not None:
            config_kwargs["thinking_budget"] = request.thinking_budget
        if request.extra_body:
            config_kwargs.update(request.extra_body)

        gen_config = self._types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        resp = self._client.models.generate_content(
            model=request.model,
            contents=self._to_contents(non_system_msgs),
            config=gen_config,
        )
        latency = time.perf_counter() - start

        usage_meta = getattr(resp, "usage_metadata", None)
        usage = usage_dict(
            getattr(usage_meta, "prompt_token_count", None),
            getattr(usage_meta, "candidates_token_count", None),
        )

        text = getattr(resp, "text", "") or ""
        request_id = getattr(resp, "response_id", None) or request.request_id
        raw = None
        if include_raw:
            if hasattr(resp, "model_dump"):
                raw = resp.model_dump(exclude_none=True)
            elif isinstance(resp, dict):
                raw = resp

        return LLMResponse(
            provider=self.provider_name,
            model=request.model,
            text=text,
            usage=usage,
            latency_s=latency,
            request_id=request_id,
            raw_response=raw,
        )
