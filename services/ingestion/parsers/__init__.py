"""Ingestion parsers package.

Avoid eager imports here to prevent circular import issues when importing
submodules like ``parsers.docx_parser``. Expose common names lazily via
``__getattr__`` while still supporting ``from parsers import X`` and static
analysis tools through ``__all__``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseParser",
    "ParserError",
    "RawBlock",
    "DocxParser",
    "EbookParser",
    "PdfParser",
    "OcrCache",
    "OcrEngine",
    "merge_ocr_text",
    "PptxParser",
    "ScormParser",
    "XlsxParser",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple import proxy
    if name in {"BaseParser", "ParserError", "RawBlock"}:
        from .base import BaseParser, ParserError, RawBlock

        mapping = {
            "BaseParser": BaseParser,
            "ParserError": ParserError,
            "RawBlock": RawBlock,
        }
        return mapping[name]

    if name == "DocxParser":
        from .docx_parser import DocxParser

        return DocxParser

    if name == "EbookParser":
        from .ebook_parser import EbookParser

        return EbookParser

    if name in {"OcrCache", "OcrEngine", "merge_ocr_text"}:
        from .ocr import OcrCache, OcrEngine, merge_ocr_text

        mapping = {
            "OcrCache": OcrCache,
            "OcrEngine": OcrEngine,
            "merge_ocr_text": merge_ocr_text,
        }
        return mapping[name]

    if name == "PdfParser":
        from .pdf_parser import PdfParser

        return PdfParser

    if name == "PptxParser":
        from .pptx_parser import PptxParser

        return PptxParser

    if name == "ScormParser":
        from .scorm_parser import ScormParser

        return ScormParser

    if name == "XlsxParser":
        from .xlsx_parser import XlsxParser

        return XlsxParser

    raise AttributeError(name)