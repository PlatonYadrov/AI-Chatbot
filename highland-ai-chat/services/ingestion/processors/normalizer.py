"""Text normalisation helpers."""

from __future__ import annotations

import re
from typing import Iterable

CONTROL_CHARS = re.compile(r"[\u0000-\u0008\u000b-\u000c\u000e-\u001f\u007f]")
MULTISPACE = re.compile(r"\s+")
FOOTER_HEADER = re.compile(r"^((стр\.|page)\s*\d+|confidential.*)$", re.IGNORECASE)


def normalise_text(raw_text: str) -> str:
    """Clean up raw text emitted by parsers.

    The function removes control characters, collapses whitespace, strips common
    footers/headers and replaces non-breaking spaces.  These steps mirror the
    practices documented in the practical RAG guide and ensures consistent
    inputs for downstream chunking.
    """

    text = raw_text.replace("\r", "\n").replace("\u00a0", " ")
    text = CONTROL_CHARS.sub(" ", text)
    cleaned_lines = []
    for line in text.splitlines():
        stripped = MULTISPACE.sub(" ", line).strip()
        if not stripped or FOOTER_HEADER.match(stripped):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def normalise_blocks(blocks: Iterable[str]) -> list[str]:
    """Apply :func:`normalise_text` to a collection preserving order."""

    return [normalise_text(block) for block in blocks if block]


__all__ = ["normalise_text", "normalise_blocks"]
