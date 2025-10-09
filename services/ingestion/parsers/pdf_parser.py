"""PDF parsing utilities built on top of PyMuPDF and pdfplumber."""

from __future__ import annotations

import logging
import re
from typing import Iterator, Optional, Tuple

from .base import BaseParser, RawBlock, ensure_dependency
from .ocr import OcrEngine, merge_ocr_text

LOGGER = logging.getLogger(__name__)


class PdfParser(BaseParser):
    """Extract text blocks and tables from PDF documents."""

    def __init__(
        self,
        *,
        extract_tables: bool = True,
        ocr_engine: Optional[OcrEngine] = None,
    ) -> None:
        self.extract_tables = extract_tables
        self.ocr_engine = ocr_engine or OcrEngine()

    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        ensure_dependency("fitz", "pip install pymupdf")
        import fitz  # type: ignore

        resolved_id = self._resolve_doc_id(path, doc_id)
        LOGGER.debug("Parsing PDF document", extra={"doc_id": resolved_id, "path": path})

        document = fitz.open(path)
        try:
            for page_index, page in enumerate(document, start=1):
                blocks = list(page.get_text("blocks"))
                text_blocks: list[RawBlock] = []
                for raw in blocks:
                    if len(raw) < 5:
                        continue
                    x0, y0, x1, y1, text, *_ = raw
                    normalised = self._clean_block(text)
                    if not normalised:
                        continue
                    text_blocks.append(
                        RawBlock(
                            text=normalised,
                            meta={
                                "doc_id": resolved_id,
                                "type": "pdf",
                                "path": path,
                                "page": page_index,
                                "bbox": (x0, y0, x1, y1),
                            },
                        )
                    )

                if self.ocr_engine.is_available():
                    text_blocks = self._attach_image_texts(page, text_blocks, path, resolved_id, page_index)

                for block in text_blocks:
                    yield block
        finally:
            document.close()

        if self.extract_tables:
            yield from self._parse_tables(path, resolved_id)

    def _parse_tables(self, path: str, doc_id: str) -> Iterator[RawBlock]:
        ensure_dependency("pdfplumber", "pip install pdfplumber")
        import pdfplumber  # type: ignore
        import pandas as pd

        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    if not table:
                        continue
                    header, *rows = table
                    if not header or not rows:
                        continue
                    frame = pd.DataFrame(rows, columns=header)
                    text = frame.to_markdown(index=False)
                    yield RawBlock(
                        text=text,
                        meta={
                            "doc_id": doc_id,
                            "type": "pdf_table",
                            "path": path,
                            "page": page_index,
                        },
                    )

    @staticmethod
    def _clean_block(text: str) -> str:
        text = text or ""
        text = text.replace("\r", "\n")
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\s+\n", "\n", text)
        return text.strip()

    def _attach_image_texts(
        self,
        page,
        blocks: list[RawBlock],
        path: str,
        doc_id: str,
        page_index: int,
    ) -> list[RawBlock]:
        raw_dict = page.get_text("rawdict")
        images: list[Tuple[bytes, Tuple[float, float, float, float]]] = []
        for element in raw_dict.get("blocks", []):
            if element.get("type") != 1:
                continue
            xref = element.get("image")
            if not xref:
                continue
            pix = page.parent.extract_image(xref)  # type: ignore[attr-defined]
            if not pix:
                continue
            images.append((pix.get("image", b""), tuple(element.get("bbox", (0, 0, 0, 0)))))

        if not images:
            return blocks

        for image_bytes, bbox in images:
            ocr_text = self.ocr_engine.extract_text(image_bytes)
            if not ocr_text:
                continue
            target = _closest_block(blocks, bbox)
            if target is not None:
                target.text = merge_ocr_text(target.text, ocr_text=ocr_text)
                target.meta["has_ocr"] = True
                target.meta["image_count"] = target.meta.get("image_count", 0) + 1
            else:
                blocks.append(
                    RawBlock(
                        text=merge_ocr_text("", ocr_text=ocr_text),
                        meta={
                            "doc_id": doc_id,
                            "type": "pdf",
                            "path": path,
                            "page": page_index,
                            "has_ocr": True,
                            "image_count": 1,
                            "bbox": bbox,
                        },
                    )
                )
        return blocks


__all__ = ["PdfParser"]


def _closest_block(blocks: list[RawBlock], bbox: Tuple[float, float, float, float]) -> Optional[RawBlock]:
    if not blocks:
        return None
    bx = (bbox[0] + bbox[2]) / 2
    by = (bbox[1] + bbox[3]) / 2
    best: Optional[RawBlock] = None
    best_distance = float("inf")
    for block in blocks:
        block_bbox = block.meta.get("bbox")
        if not block_bbox:
            continue
        cx = (block_bbox[0] + block_bbox[2]) / 2
        cy = (block_bbox[1] + block_bbox[3]) / 2
        distance = (cx - bx) ** 2 + (cy - by) ** 2
        if distance < best_distance:
            best = block
            best_distance = distance
    return best