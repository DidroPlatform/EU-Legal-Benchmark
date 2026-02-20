from __future__ import annotations

import threading
import time
from typing import Callable

from src.types import WaitableRateLimiter


class PerMinuteRateLimiter(WaitableRateLimiter):
    def __init__(
        self,
        requests_per_minute: int,
        monotonic_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self._interval_s = 60.0 / float(requests_per_minute)
        self._next_allowed = 0.0
        self._lock = threading.Lock()
        self._monotonic = monotonic_fn or time.monotonic
        self._sleep = sleep_fn or time.sleep

    def wait(self) -> None:
        with self._lock:
            now = self._monotonic()
            wait_for = max(0.0, self._next_allowed - now)
            scheduled = max(self._next_allowed, now) + self._interval_s
            self._next_allowed = scheduled
        if wait_for > 0:
            self._sleep(wait_for)
