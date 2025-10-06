"""
Маскирование PII: простые regex и заглушка для NER.
Вход: документы; Выход: документы с замаскированной PII.
"""

from __future__ import annotations

import re
from typing import Any, Dict


PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{2}\b"),  # пример номера
    re.compile(r"\b\+7\d{10}\b"),  # телефон РФ
]


def redact(text: str) -> str:
    out = text or ""
    for pat in PII_PATTERNS:
        out = pat.sub("[PII]", out)
    return out


def process(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    doc["text"] = redact(doc.get("text", ""))
    return doc


__all__ = ["process", "redact"]

