"""Ingestion parsers package.

Unified document parsing using Docling library.
Legacy parsers have been replaced by DoclingParser.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseParser",
    "ParserError",
    "RawBlock",
    "DoclingParser",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseParser", "ParserError", "RawBlock"}:
        from .base import BaseParser, ParserError, RawBlock

        mapping = {
            "BaseParser": BaseParser,
            "ParserError": ParserError,
            "RawBlock": RawBlock,
        }
        return mapping[name]

    if name == "DoclingParser":
        from .docling_parser import DoclingParser

        return DoclingParser

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")