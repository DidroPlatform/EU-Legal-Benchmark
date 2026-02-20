from __future__ import annotations

import unittest
from unittest import mock

from src.config import RetryConfig
from src.retry import is_transient_error, with_retries


class TestRetry(unittest.TestCase):
    def test_is_transient_error_from_status_code(self) -> None:
        err = Exception("any")
        setattr(err, "status_code", 503)
        self.assertTrue(is_transient_error(err))

    def test_is_transient_error_from_response_status_code(self) -> None:
        class FakeResponse:
            status_code = 504

        err = Exception("wrapped")
        setattr(err, "response", FakeResponse())
        self.assertTrue(is_transient_error(err))

    def test_is_transient_error_detects_capacity_markers(self) -> None:
        err = Exception("service_tier_capacity_exceeded while overloaded")
        self.assertTrue(is_transient_error(err))

    def test_is_transient_error_false_for_non_transient(self) -> None:
        err = Exception("validation failed: malformed input")
        self.assertFalse(is_transient_error(err))

    def test_before_attempt_runs_for_each_attempt(self) -> None:
        attempts: list[int] = []
        call_count = 0

        def before_attempt(attempt: int) -> None:
            attempts.append(attempt)

        def flaky_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 temporary")
            return "ok"

        with (
            mock.patch("src.retry.time.sleep", return_value=None),
            mock.patch("src.retry.random.uniform", return_value=0.0),
        ):
            result = with_retries(
                flaky_call,
                RetryConfig(max_attempts=5, base_delay_s=1.0, max_delay_s=30.0),
                before_attempt=before_attempt,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, [1, 2, 3])

    def test_retry_after_header_takes_precedence_and_is_capped(self) -> None:
        call_count = 0

        class FakeResponse:
            headers = {"Retry-After": "99"}

        err = Exception("429 too many requests")
        setattr(err, "response", FakeResponse())

        def flaky_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise err
            return "ok"

        with (
            mock.patch("src.retry.time.sleep", return_value=None) as sleep_mock,
            mock.patch("src.retry.random.uniform", return_value=0.0),
        ):
            result = with_retries(
                flaky_call,
                RetryConfig(max_attempts=3, base_delay_s=1.0, max_delay_s=30.0),
            )

        self.assertEqual(result, "ok")
        sleep_mock.assert_called_once_with(30.0)

