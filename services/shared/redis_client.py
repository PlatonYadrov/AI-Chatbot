from __future__ import annotations

import os
from typing import Optional


class RedisClient:
    """Простая заглушка клиента Redis (без внешних зависимостей)."""

    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or os.getenv("REDIS_URL", "redis://redis:6379/0")

    # Заглушки API
    def get(self, key: str) -> Optional[str]:  # type: ignore[override]
        return None

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        return None

    def publish(self, channel: str, message: str) -> None:
        return None


def get_redis() -> RedisClient:
    return RedisClient()


__all__ = ["RedisClient", "get_redis"]

