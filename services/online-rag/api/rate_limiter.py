from __future__ import annotations

import time
from typing import Callable

from fastapi import HTTPException, Request, status


class TokenBucket:
    def __init__(self, rate_per_minute: int, capacity: int | None = None) -> None:
        self.rate = rate_per_minute / 60.0
        self.capacity = capacity or rate_per_minute
        self.tokens = float(self.capacity)
        self.timestamp = time.time()

    def allow(self, now: float | None = None) -> bool:
        now = now or time.time()
        elapsed = max(0.0, now - self.timestamp)
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


def limiter(rate_per_minute: int) -> Callable[[Request], None]:
    bucket = TokenBucket(rate_per_minute)

    def dependency(_: Request) -> None:
        if not bucket.allow():
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded")

    return dependency


__all__ = ["limiter", "TokenBucket"]

