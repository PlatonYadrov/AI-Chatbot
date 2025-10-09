"""Parse Microsoft Excel workbooks into logical row windows."""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from .base import BaseParser, RawBlock, ensure_dependency
from .ocr import OcrEngine, merge_ocr_text

LOGGER = logging.getLogger(__name__)


class XlsxParser(BaseParser):
    """Parse Microsoft Excel workbooks into logical row windows."""

    def __init__(self, *, window: int = 50, ocr_engine: Optional[OcrEngine] = None) -> None:
        self.window = max(1, window)
        self.ocr_engine = ocr_engine or OcrEngine()

    def parse(self, path: str, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        ensure_dependency("pandas", "pip install pandas openpyxl")
        import pandas as pd

        resolved_id = self._resolve_doc_id(path, doc_id)
        LOGGER.debug("Parsing XLSX document", extra={"doc_id": resolved_id, "path": path})

        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            frame = xls.parse(sheet)
            if frame.empty:
                continue

            summary = (
                f"Sheet: {sheet}\nColumns: {', '.join(map(str, frame.columns))}\n"
                f"Rows: {len(frame)}"
            )
            yield RawBlock(
                text=summary,
                meta={
                    "doc_id": resolved_id,
                    "type": "xlsx_sheet",
                    "path": path,
                    "sheet": sheet,
                },
            )

            for start in range(0, len(frame), self.window):
                chunk = frame.iloc[start : start + self.window]
                text = chunk.to_markdown(index=False)
                kv_lines = "\n".join(
                    " | ".join(f"{col}: {row[col]}" for col in chunk.columns)
                    for _, row in chunk.iterrows()
                )
                yield RawBlock(
                    text=f"{text}\n\n{kv_lines}",
                    meta={
                        "doc_id": resolved_id,
                        "type": "xlsx_rows",
                        "path": path,
                        "sheet": sheet,
                        "row_range": (start, min(start + self.window, len(frame))),
                    },
                )

            yield from self._extract_images(path, resolved_id, sheet)

    def _extract_images(self, path: str, doc_id: str, sheet: str) -> Iterator[RawBlock]:
        if not self.ocr_engine.is_available():
            return

        try:
            ensure_dependency("openpyxl", "pip install openpyxl")
            from openpyxl import load_workbook  # type: ignore
        except Exception:  # pragma: no cover - optional path for missing dependency
            return

        workbook = load_workbook(path)
        worksheet = workbook[sheet]
        images = getattr(worksheet, "_images", []) or []
        for image in images:
            blob = getattr(image, "_data", None)
            if not blob:
                continue
            text = self.ocr_engine.extract_text(blob()) if callable(blob) else self.ocr_engine.extract_text(blob)
            if not text:
                continue
            yield RawBlock(
                text=merge_ocr_text("", ocr_text=text),
                meta={
                    "doc_id": doc_id,
                    "type": "xlsx_image",
                    "path": path,
                    "sheet": sheet,
                    "has_ocr": True,
                },
            )


__all__ = ["XlsxParser"]