from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from .base import BaseProvider, usage_dict
from ..config import ProviderConfig
from ..types import LLMRequest, LLMResponse


_LITELLM_FINISH_REASON_SHIM_INSTALLED = False
_LITELLM_FINISH_REASON_SHIM_LOCK = threading.Lock()


def _install_litellm_finish_reason_compat_shim() -> None:
    global _LITELLM_FINISH_REASON_SHIM_INSTALLED
    if _LITELLM_FINISH_REASON_SHIM_INSTALLED:
        return

    with _LITELLM_FINISH_REASON_SHIM_LOCK:
        if _LITELLM_FINISH_REASON_SHIM_INSTALLED:
            return

        patched_any = False
        module_paths = (
            "litellm.types.utils",
            "litellm.litellm_core_utils.core_helpers",
        )
        for module_path in module_paths:
            try:
                module = __import__(module_path, fromlist=["map_finish_reason"])
            except Exception:
                continue

            original = getattr(module, "map_finish_reason", None)
            if not callable(original):
                continue
            if getattr(original, "_legal_benchmark_finish_reason_shim", False):
                patched_any = True
                continue

            def _map_finish_reason_compat_wrapper(
                finish_reason: str, _original=original
            ):  # pragma: no cover - exercised via patched modules in tests
                mapped_reason = "content_filter" if finish_reason == "content_filtered" else finish_reason
                return _original(mapped_reason)

            setattr(_map_finish_reason_compat_wrapper, "_legal_benchmark_finish_reason_shim", True)
            setattr(module, "map_finish_reason", _map_finish_reason_compat_wrapper)
            patched_any = True

        if patched_any:
            _LITELLM_FINISH_REASON_SHIM_INSTALLED = True


class LiteLLMProvider(BaseProvider):
    def __init__(self, provider_name: str, config: ProviderConfig, api_key: str | None):
        super().__init__(provider_name, config, api_key)
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install dependency 'litellm' to use LiteLLM provider.") from exc

        self._completion = completion
        _install_litellm_finish_reason_compat_shim()
        self._timeout_s = config.timeout_s
        self._base_url = config.base_url

    def _resolve_model(self, model: str) -> str:
        resolved = model.strip()
        if self.provider_name == "vertex" and not resolved.startswith("vertex_ai/"):
            # Keep Vertex routing explicit and stable across LiteLLM defaults.
            return f"vertex_ai/{resolved}"
        if self.provider_name == "anthropic_vertex" and not resolved.startswith("vertex_ai/"):
            # Anthropic on Vertex must route through LiteLLM's vertex_ai provider.
            return f"vertex_ai/{resolved}"
        return resolved

    def generate(self, request: LLMRequest, include_raw: bool = False) -> LLMResponse:
        start = time.perf_counter()
        messages: List[Dict[str, str]] = [{"role": m.role, "content": m.content} for m in request.messages]
        resolved_model = self._resolve_model(request.model)

        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": request.temperature,
            "timeout": self._timeout_s,
        }

        # Bedrock Claude rejects requests that include both temperature and top_p.
        # Prefer temperature when both are set to preserve current config behavior.
        omit_top_p = (
            resolved_model.startswith("bedrock/")
            and "anthropic.claude" in resolved_model
            and request.top_p is not None
            and request.temperature is not None
        )
        if request.top_p is not None and not omit_top_p:
            kwargs["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.extra_body:
            if self.provider_name in {"vertex", "anthropic_vertex"}:
                kwargs.update(request.extra_body)
            else:
                kwargs["extra_body"] = request.extra_body
        if self._base_url:
            kwargs["api_base"] = self._base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.provider_name in {"vertex", "anthropic_vertex"}:
            if self.config.project:
                kwargs["vertex_project"] = self.config.project
            if self.config.location:
                kwargs["vertex_location"] = self.config.location

        resp = self._completion(**kwargs)
        latency = time.perf_counter() - start

        if hasattr(resp, "model_dump"):
            resp_dict = resp.model_dump()
        elif isinstance(resp, dict):
            resp_dict = resp
        else:
            resp_dict = {}

        choices = resp_dict.get("choices", [])
        text = ""
        if choices:
            message = (choices[0] or {}).get("message", {})
            text = message.get("content", "") or ""

        usage_obj = resp_dict.get("usage") or {}
        usage = usage_dict(
            usage_obj.get("prompt_tokens"),
            usage_obj.get("completion_tokens"),
        )

        request_id = resp_dict.get("id") or request.request_id
        raw = resp_dict if include_raw else None

        return LLMResponse(
            provider=self.provider_name,
            model=request.model,
            text=text,
            usage=usage,
            latency_s=latency,
            request_id=request_id,
            raw_response=raw,
        )
