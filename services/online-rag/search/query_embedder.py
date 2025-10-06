from __future__ import annotations

import hashlib
import json
from typing import List

from ...shared.redis_client import get_redis


def _hash_key(text: str) -> str:
    h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    return f"emb:{h}"


def _fake_embed(text: str) -> List[float]:
    h = hashlib.sha1((text or "").encode("utf-8")).digest()
    return [int(x) / 255.0 for x in h[:8]]


def encode(text: str) -> List[float]:
    cache = get_redis()
    key = _hash_key(text)
    cached = cache.get(key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:  # noqa: BLE001
            pass
    vec = _fake_embed(text)
    try:
        cache.set(key, json.dumps(vec), ex=86400)
    except Exception:
        pass
    return vec


__all__ = ["encode"]

