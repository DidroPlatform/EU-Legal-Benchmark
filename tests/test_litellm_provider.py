from __future__ import annotations

import types
import unittest
from unittest import mock

import src.providers.litellm as litellm_provider_module
from src.config import ProviderConfig
from src.providers.litellm import LiteLLMProvider
from src.types import LLMMessage, LLMRequest


class TestLiteLLMProvider(unittest.TestCase):
    def setUp(self) -> None:
        litellm_provider_module._LITELLM_FINISH_REASON_SHIM_INSTALLED = False

    @staticmethod
    def _request(model: str) -> LLMRequest:
        return LLMRequest(
            provider="bedrock",
            model=model,
            messages=[LLMMessage(role="user", content="hello")],
            temperature=0.2,
            top_p=1.0,
            request_id="req-1",
        )

    def test_bedrock_claude_omits_top_p_when_temperature_set(self) -> None:
        captured: dict[str, object] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "id": "resp-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        with mock.patch.dict("sys.modules", {"litellm": types.SimpleNamespace(completion=fake_completion)}):
            provider = LiteLLMProvider("bedrock", ProviderConfig(timeout_s=5), api_key=None)

            provider.generate(
                self._request(
                    "bedrock/arn:aws:bedrock:eu-north-1:123:inference-profile/eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
                )
            )

        self.assertIn("temperature", captured)
        self.assertNotIn("top_p", captured)

    def test_non_claude_requests_keep_top_p(self) -> None:
        captured: dict[str, object] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "id": "resp-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        with mock.patch.dict("sys.modules", {"litellm": types.SimpleNamespace(completion=fake_completion)}):
            provider = LiteLLMProvider("bedrock", ProviderConfig(timeout_s=5), api_key=None)

            provider.generate(self._request("bedrock/converse/openai.gpt-oss-120b-1:0"))

        self.assertIn("temperature", captured)
        self.assertIn("top_p", captured)

    def test_finish_reason_shim_maps_content_filtered(self) -> None:
        litellm_module = types.ModuleType("litellm")
        litellm_types_module = types.ModuleType("litellm.types")
        litellm_types_utils_module = types.ModuleType("litellm.types.utils")
        litellm_core_module = types.ModuleType("litellm.litellm_core_utils")
        litellm_core_helpers_module = types.ModuleType("litellm.litellm_core_utils.core_helpers")

        def fake_completion(**kwargs):
            return {
                "id": "resp-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        def fake_map_finish_reason(value: str) -> str:
            return value

        litellm_module.completion = fake_completion
        litellm_types_utils_module.map_finish_reason = fake_map_finish_reason
        litellm_core_helpers_module.map_finish_reason = fake_map_finish_reason

        with mock.patch.dict(
            "sys.modules",
            {
                "litellm": litellm_module,
                "litellm.types": litellm_types_module,
                "litellm.types.utils": litellm_types_utils_module,
                "litellm.litellm_core_utils": litellm_core_module,
                "litellm.litellm_core_utils.core_helpers": litellm_core_helpers_module,
            },
        ):
            LiteLLMProvider("bedrock", ProviderConfig(timeout_s=5), api_key=None)

            self.assertEqual(litellm_types_utils_module.map_finish_reason("content_filtered"), "content_filter")
            self.assertEqual(litellm_core_helpers_module.map_finish_reason("content_filtered"), "content_filter")
            self.assertEqual(litellm_types_utils_module.map_finish_reason("stop"), "stop")


if __name__ == "__main__":
    unittest.main()
