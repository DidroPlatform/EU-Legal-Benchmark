from __future__ import annotations

import os
import unittest
from unittest import mock

from src.config import BenchmarkConfig, DataConfig, DatasetConfig, ModelConfig, ProviderConfig, RunConfig
from src.setup_checks import check_setup, required_provider_names


class TestSetupProviderProfiles(unittest.TestCase):
    def _build_config(self) -> BenchmarkConfig:
        candidates = [
            ModelConfig(name="c_nim", provider="nim", model="moonshotai/kimi-k2.5"),
            ModelConfig(name="c_bedrock", provider="bedrock", model="bedrock/anthropic.claude-opus-4-6-v1"),
            ModelConfig(name="c_mistral", provider="mistral_api", model="mistral/mistral-large-latest"),
            ModelConfig(
                name="c_vercel",
                provider="vercel_gateway",
                model="vercel_ai_gateway/openai/gpt-5.2",
            ),
        ]
        judge = ModelConfig(name="j_gemini", provider="google_genai", model="gemini-flash-lite-latest")
        return BenchmarkConfig(
            providers={
                "nim": ProviderConfig(api_key_env="NVIDIA_API_KEY", base_url="https://integrate.api.nvidia.com/v1"),
                "bedrock": ProviderConfig(api_key_env="AWS_BEARER_TOKEN_BEDROCK"),
                "mistral_api": ProviderConfig(api_key_env="MISTRAL_API_KEY"),
                "vercel_gateway": ProviderConfig(
                    api_key_env="AI_GATEWAY_API_KEY",
                    base_url="https://ai-gateway.vercel.sh/v1",
                ),
                "google_genai": ProviderConfig(api_key_env="GEMINI_API_KEY"),
            },
            candidates=candidates,
            judge=judge,
            judges=[judge],
            data=DataConfig(
                datasets=[
                    DatasetConfig(
                        name="d",
                        path=__file__,  # Existing file path to satisfy config validation.
                        provenance="canonical:test",
                        judge_mode="reference",
                    )
                ]
            ),
            run=RunConfig(response_rate_limit_rpm=50),
        )

    def test_required_provider_names_includes_all_profiles(self) -> None:
        cfg = self._build_config()
        self.assertEqual(
            required_provider_names(cfg),
            {"nim", "bedrock", "mistral_api", "vercel_gateway", "google_genai"},
        )

    def test_check_setup_reports_missing_env_vars_for_all_profiles(self) -> None:
        cfg = self._build_config()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("src.setup_checks.importlib.util.find_spec", return_value=object()),
        ):
            report = check_setup(cfg)

        joined = "\n".join(report.errors)
        self.assertIn("Provider 'nim' requires env var 'NVIDIA_API_KEY'", joined)
        self.assertIn("Provider 'bedrock' requires env var 'AWS_BEARER_TOKEN_BEDROCK'", joined)
        self.assertIn("Provider 'mistral_api' requires env var 'MISTRAL_API_KEY'", joined)
        self.assertIn("Provider 'vercel_gateway' requires env var 'AI_GATEWAY_API_KEY'", joined)
        self.assertIn("Provider 'google_genai' requires env var 'GEMINI_API_KEY'", joined)

    def test_check_setup_reports_missing_boto3_for_bedrock_models(self) -> None:
        cfg = self._build_config()

        def fake_find_spec(name: str):
            if name == "boto3":
                return None
            return object()

        with (
            mock.patch.dict(
                os.environ,
                {
                    "NVIDIA_API_KEY": "x",
                    "AWS_BEARER_TOKEN_BEDROCK": "x",
                    "MISTRAL_API_KEY": "x",
                    "AI_GATEWAY_API_KEY": "x",
                    "GEMINI_API_KEY": "x",
                },
                clear=True,
            ),
            mock.patch("src.setup_checks.importlib.util.find_spec", side_effect=fake_find_spec),
        ):
            report = check_setup(cfg)

        joined = "\n".join(report.errors)
        self.assertIn("Bedrock model routing requires Python package 'boto3'", joined)

    def test_check_setup_does_not_require_boto3_without_bedrock_models(self) -> None:
        cfg = self._build_config()
        cfg.candidates = [
            candidate
            for candidate in cfg.candidates
            if not candidate.model.strip().startswith("bedrock/")
        ]

        def fake_find_spec(name: str):
            if name == "boto3":
                return None
            return object()

        with (
            mock.patch.dict(
                os.environ,
                {
                    "NVIDIA_API_KEY": "x",
                    "MISTRAL_API_KEY": "x",
                    "AI_GATEWAY_API_KEY": "x",
                    "GEMINI_API_KEY": "x",
                },
                clear=True,
            ),
            mock.patch("src.setup_checks.importlib.util.find_spec", side_effect=fake_find_spec),
        ):
            report = check_setup(cfg)

        self.assertTrue(report.ok, msg="\n".join(report.errors))


if __name__ == "__main__":
    unittest.main()
