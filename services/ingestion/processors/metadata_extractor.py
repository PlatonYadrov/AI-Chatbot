"""
Извлечение метаданных: категории, manager_type, версии, effective_* из текста/мета.
Вход: parsed_docs; Выход: обогащённые документы (payload заполнен).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def enrich(doc: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "category": doc.get("source_meta", {}).get("category", "general"),
        "manager_type": doc.get("source_meta", {}).get("manager_type", "line"),
        "source": doc.get("source_meta", {}).get("source"),
        "version": doc.get("source_meta", {}).get("version", datetime.utcnow().strftime("v%Y.%m.%d")),
        "effective_from": doc.get("source_meta", {}).get("effective_from"),
        "effective_to": doc.get("source_meta", {}).get("effective_to"),
        "acl": doc.get("source_meta", {}).get("acl", []),
        "section": doc.get("source_meta", {}).get("section"),
        "language": doc.get("source_meta", {}).get("language", "ru"),
    }
    out = dict(doc)
    out["source_meta"] = payload
    return out


__all__ = ["enrich"]

