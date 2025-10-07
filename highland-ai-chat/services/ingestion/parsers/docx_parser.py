"""DOCX parser implementation with optional OCR support."""

from __future__ import annotations

import logging
import re
from typing import Iterator, Optional

from .base import BaseParser, RawBlock, ensure_dependency
from .ocr import OcrEngine, merge_ocr_text

LOGGER = logging.getLogger(__name__)


class DocxParser(BaseParser):
    """Parse DOCX files into :class:`RawBlock` streams.

    The parser keeps track of heading hierarchy so that downstream components
    can reconstruct the section context of each block.  Heading detection relies
    on the canonical Microsoft Office style naming scheme but also supports the
    Russian localisation used in many enterprises.
    """

    HEADING_RE = re.compile(r"(heading|заголовок)\s*(?P<level>\d+)", re.IGNORECASE)

    def __init__(self, *, ocr_engine: Optional[OcrEngine] = None) -> None:
        self.ocr_engine = ocr_engine or OcrEngine()

    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        ensure_dependency("docx", "pip install python-docx")
        from docx import Document  # type: ignore

        resolved_id = self._resolve_doc_id(path, doc_id)
        LOGGER.debug("Parsing DOCX document", extra={"doc_id": resolved_id, "path": path})

        document = Document(path)
        heading_stack: list[str] = []
        blocks: list[RawBlock] = []

        for paragraph in document.paragraphs:
            text = self._normalise(paragraph.text)
            if not text:
                continue

            style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
            level = self._extract_level(style_name)
            if level:
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(text)
                blocks.append(
                    RawBlock(
                        text=text,
                        meta={
                            "doc_id": resolved_id,
                            "type": "docx",
                            "path": path,
                            "headings": heading_stack.copy(),
                            "is_heading": True,
                        },
                    )
                )
                continue

            blocks.append(
                RawBlock(
                    text=text,
                    meta={
                        "doc_id": resolved_id,
                        "type": "docx",
                        "path": path,
                        "headings": heading_stack.copy(),
                    },
                )
            )

        enriched_blocks = self._inject_images(document, blocks, path, resolved_id)
        for block in enriched_blocks:
            yield block

    def _extract_level(self, style: str) -> Optional[int]:
        if not style:
            return None
        match = self.HEADING_RE.search(style)
        if not match:
            return None
        try:
            return int(match.group("level"))
        except ValueError:  # pragma: no cover - defensive
            LOGGER.debug("Failed to parse heading level", extra={"style": style})
            return None

    def _inject_images(
        self,
        document,
        blocks: list[RawBlock],
        path: str,
        doc_id: str,
    ) -> Iterator[RawBlock]:
        if not self.ocr_engine.is_available():
            yield from blocks
            return

        if not getattr(document.part, "rels", None):
            yield from blocks
            return

        last_index = len(blocks) - 1
        for rel in document.part.rels.values():
            target = getattr(rel, "target_ref", "") or ""
            if "image" not in target:
                continue
            image_part = getattr(rel, "_target", None)
            blob = getattr(image_part, "blob", None)
            if not blob:
                continue
            text = self.ocr_engine.extract_text(blob)
            if not text:
                continue
            if last_index >= 0:
                block = blocks[last_index]
                block.text = merge_ocr_text(block.text, ocr_text=text)
                block.meta["has_ocr"] = True
                block.meta["image_count"] = block.meta.get("image_count", 0) + 1
            else:
                blocks.append(
                    RawBlock(
                        text=merge_ocr_text("", ocr_text=text),
                        meta={
                            "doc_id": doc_id,
                            "type": "docx",
                            "path": path,
                            "has_ocr": True,
                            "image_count": 1,
                        },
                    )
                )
                last_index = len(blocks) - 1

        yield from blocks


__all__ = ["DocxParser"]
