from __future__ import annotations

from typing import Any, Dict, List


def validate(answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Заглушка entailment/anti-PII: всегда ok
    return {"ok": True, "reason": None}


__all__ = ["validate"]

