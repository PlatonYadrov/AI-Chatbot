"""Metadata extraction primitives."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict

DATE_RE = re.compile(r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})")
VERSION_RE = re.compile(r"v(?:ersion)?\s*(\d+[._]\d+(?:[._]\d+)*)", re.IGNORECASE)
LANG_RE = re.compile(r"\b(ru|en|de|fr)\b", re.IGNORECASE)


def enrich_metadata(chunk: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Enrich chunk level metadata using simple rule-based heuristics."""

    metadata = dict(chunk)
    if "version" not in metadata:
        version = _extract_version(text)
        if version:
            metadata["version"] = version

    if "language" not in metadata:
        language = _extract_language(text)
        if language:
            metadata["language"] = language

    if "effective_from" not in metadata:
        effective = _extract_date(text)
        if effective:
            metadata["effective_from"] = effective
    return metadata


def _extract_version(text: str) -> str | None:
    match = VERSION_RE.search(text)
    if match:
        return match.group(1)
    return None


def _extract_language(text: str) -> str | None:
    match = LANG_RE.search(text)
    if match:
        return match.group(1).lower()
    return None


def _extract_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    for pattern in ("%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(match.group(1), pattern).date().isoformat()
        except ValueError:
            continue
    return None


__all__ = ["enrich_metadata"]
