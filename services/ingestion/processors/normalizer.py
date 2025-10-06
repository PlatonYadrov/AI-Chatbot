"""
Нормализация текста: unicode нормализация, удаление мусора, стандартизация пробелов.
Вход: parsed_docs; Выход: normalized_docs (та же схема, текст очищен).
SLO: ≥95% чанков за < 1 s.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    doc["text"] = normalize_text(doc.get("text", ""))
    return doc


__all__ = ["process", "normalize_text"]

