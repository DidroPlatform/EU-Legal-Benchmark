from __future__ import annotations

import re
import random
import time
from typing import Callable, Optional, TypeVar

from .config import RetryConfig


T = TypeVar("T")


TRANSIENT_MARKERS = (
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection",
    "connecttimeout",
    "connecterror",
    "remoteprotocolerror",
    "disconnected",
    "unavailable",
    "deadline_exceeded",
    "ratelimiterror",
    "service_tier_capacity_exceeded",
    "capacity exceeded",
    "overloaded",
    "empty response text",
    "service unavailable",
    "internal server error",
)


def _extract_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _extract_status_text(exc: Exception) -> str:
    value = getattr(exc, "status", None)
    if value is None:
        return ""
    return str(value).lower()


def is_transient_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    status_text = _extract_status_text(exc)
    if status_text in {"unavailable", "deadline_exceeded", "service_unavailable", "too_many_requests"}:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in TRANSIENT_MARKERS)


def _parse_retry_after_value(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        retry_after = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if retry_after < 0:
        return None
    return retry_after


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    retry_after_attr = getattr(exc, "retry_after", None)
    parsed_attr = _parse_retry_after_value(retry_after_attr)
    if parsed_attr is not None:
        return parsed_attr

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After", "RETRY-AFTER"):
            value = headers.get(key) if hasattr(headers, "get") else None
            parsed = _parse_retry_after_value(value)
            if parsed is not None:
                return parsed

    match = re.search(r"retry[-_ ]?after[^0-9]*([0-9]+(?:\.[0-9]+)?)", str(exc), re.IGNORECASE)
    if match:
        return _parse_retry_after_value(match.group(1))
    return None


def with_retries(
    fn: Callable[[], T],
    retry: RetryConfig,
    before_attempt: Callable[[int], None] | None = None,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, retry.max_attempts + 1):
        if before_attempt is not None:
            before_attempt(attempt)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDK exceptions vary
            last_exc = exc
            if attempt >= retry.max_attempts or not is_transient_error(exc):
                raise
            retry_after = _extract_retry_after_seconds(exc)
            if retry_after is not None:
                delay = min(retry.max_delay_s, retry_after)
            else:
                delay = min(retry.max_delay_s, retry.base_delay_s * (2 ** (attempt - 1)))
                delay = delay + random.uniform(0.0, min(0.5, retry.base_delay_s))
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
