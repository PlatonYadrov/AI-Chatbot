"""PII redaction helpers using lightweight regular expressions."""

from __future__ import annotations

import re

CARD_RE = re.compile(r"\b\d{4}(?:[\s-]?\d{4}){3}\b")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s()-]{7,}\d")
ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_pii(text: str) -> str:
    """Mask common PII entities using deterministic patterns."""

    redacted = CARD_RE.sub("[CARD]", text)
    redacted = EMAIL_RE.sub("[EMAIL]", redacted)
    redacted = PHONE_RE.sub("[PHONE]", redacted)
    redacted = ID_RE.sub("[ID]", redacted)
    return redacted


__all__ = ["redact_pii"]
