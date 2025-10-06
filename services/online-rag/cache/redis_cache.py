from __future__ import annotations

import json
from typing import Any, Optional

from ...shared.redis_client import get_redis


def get(key: str) -> Optional[Any]:
    val = get_redis().get(key)
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:  # noqa: BLE001
        return None


def set(key: str, value: Any, ttl_seconds: int | None = None) -> None:
    try:
        get_redis().set(key, json.dumps(value), ex=ttl_seconds)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["get", "set"]

