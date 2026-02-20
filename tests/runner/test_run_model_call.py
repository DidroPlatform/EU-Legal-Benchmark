from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import run as run_module
from src.cache import DiskCache
from src.config import RetryConfig
from src.providers.base import BaseProvider
from src.types import LLMMessage, LLMRequest, LLMResponse


class _FakeProvider(BaseProvider):
    def __init__(self, responses: list[str]):
        super().__init__("fake", config=mock.Mock(), api_key=None)
        self._responses = list(responses)
        self.calls = 0

    def generate(self, request: LLMRequest, include_raw: bool = False) -> LLMResponse:
        self.calls += 1
        text = self._responses.pop(0)
        return LLMResponse(
            provider="fake",
            model=request.model,
            text=text,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_s=0.01,
            request_id=request.request_id,
            raw_response=None,
        )


class TestRunModelCall(unittest.TestCase):
    @staticmethod
    def _request() -> LLMRequest:
        return LLMRequest(
            provider="fake",
            model="fake/model",
            messages=[LLMMessage(role="user", content="hello")],
            request_id="req-1",
        )

    def test_cached_empty_response_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = DiskCache(td, enabled=True)
            req = self._request()
            key = cache.make_key(run_module._cache_key_payload(req, stage="response"))
            cache.set(
                key,
                {
                    "provider": "fake",
                    "model": "fake/model",
                    "text": "",
                    "usage": {},
                    "latency_s": 0.0,
                    "request_id": "cached",
                    "raw_response": None,
                },
            )
            provider = _FakeProvider(["fresh text"])

            payload, cache_hit, _ = run_module._run_model_call(
                provider,
                req,
                cache,
                RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
                stage="response",
                include_raw=False,
            )

            self.assertFalse(cache_hit)
            self.assertEqual(payload["text"], "fresh text")
            self.assertEqual(provider.calls, 1)
            self.assertEqual(cache.get(key)["text"], "fresh text")

    def test_live_empty_response_retries_and_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = DiskCache(td, enabled=True)
            provider = _FakeProvider(["", "ok"])
            req = self._request()
            with (
                mock.patch("src.retry.time.sleep", return_value=None),
                mock.patch("src.retry.random.uniform", return_value=0.0),
            ):
                payload, cache_hit, _ = run_module._run_model_call(
                    provider,
                    req,
                    cache,
                    RetryConfig(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0),
                    stage="response",
                    include_raw=False,
                )

            self.assertFalse(cache_hit)
            self.assertEqual(payload["text"], "ok")
            self.assertEqual(provider.calls, 2)

    def test_cached_non_empty_response_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = DiskCache(td, enabled=True)
            req = self._request()
            key = cache.make_key(run_module._cache_key_payload(req, stage="response"))
            cache.set(
                key,
                {
                    "provider": "fake",
                    "model": "fake/model",
                    "text": "cached text",
                    "usage": {},
                    "latency_s": 0.0,
                    "request_id": "cached",
                    "raw_response": None,
                },
            )
            provider = _FakeProvider(["should not be used"])
            payload, cache_hit, _ = run_module._run_model_call(
                provider,
                req,
                cache,
                RetryConfig(max_attempts=2, base_delay_s=0.0, max_delay_s=0.0),
                stage="response",
                include_raw=False,
            )
            self.assertTrue(cache_hit)
            self.assertEqual(payload["text"], "cached text")
            self.assertEqual(provider.calls, 0)

