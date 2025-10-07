"""Unit tests covering OCR-aware parser utilities."""

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ingestion.parsers.base import RawBlock
from services.ingestion.parsers.docx_parser import DocxParser
from services.ingestion.parsers.ocr import OcrCache, merge_ocr_text
from services.ingestion.parsers.pdf_parser import PdfParser
from services.ingestion.parsers.pptx_parser import PptxParser


class DummyOcrEngine:
    def __init__(self, text: str = "ocr-text") -> None:
        self.text = text
        self.calls = 0

    def is_available(self) -> bool:  # pragma: no cover - simple passthrough
        return True

    def extract_text(self, image_bytes: bytes) -> str:
        self.calls += 1
        return self.text


def test_merge_ocr_text_appends_sections() -> None:
    merged = merge_ocr_text("base", ocr_text="data", caption="note")
    assert "base" in merged
    assert "[Caption] note" in merged
    assert "[OCR] data" in merged


def test_ocr_cache_memoises_results() -> None:
    cache = OcrCache()
    payload = b"bytes"
    assert cache.get(payload) is None
    cache.set(payload, "value")
    assert cache.get(payload) == "value"


def test_docx_parser_injects_ocr_text() -> None:
    engine = DummyOcrEngine("image content")
    parser = DocxParser(ocr_engine=engine)
    blocks = [RawBlock(text="base", meta={"doc_id": "doc", "type": "docx"})]
    document = SimpleNamespace(
        part=SimpleNamespace(
            rels={
                "r1": SimpleNamespace(
                    target_ref="word/media/image1.png",
                    _target=SimpleNamespace(blob=b"data"),
                )
            }
        )
    )
    updated = list(parser._inject_images(document, blocks, "file.docx", "doc"))
    assert updated[0].meta["has_ocr"] is True
    assert "image content" in updated[0].text
    assert engine.calls == 1


def test_pdf_parser_attaches_ocr_blocks() -> None:
    engine = DummyOcrEngine("diagram")
    parser = PdfParser(extract_tables=False, ocr_engine=engine)
    blocks = [
        RawBlock(text="page text", meta={"bbox": (0, 0, 10, 10)})
    ]

    class DummyPage:
        def __init__(self) -> None:
            self.parent = SimpleNamespace(extract_image=lambda _: {"image": b"blob"})

        def get_text(self, mode: str):
            if mode == "rawdict":
                return {"blocks": [{"type": 1, "image": 1, "bbox": (0, 0, 2, 2)}]}
            raise AssertionError("unexpected mode")

    updated = parser._attach_image_texts(DummyPage(), blocks, "sample.pdf", "doc", 1)
    assert updated[0].meta["has_ocr"] is True
    assert "diagram" in updated[0].text


def test_pptx_parser_attach_ocr_results() -> None:
    engine = DummyOcrEngine("slide img")
    parser = PptxParser(ocr_engine=engine)

    class DummySlide:
        def __init__(self) -> None:
            self.shapes = [
                SimpleNamespace(text="hello", image=None),
                SimpleNamespace(text="", image=SimpleNamespace(blob=b"img")),
            ]

    enriched = parser._attach_ocr(DummySlide(), "hello")
    assert "slide img" in enriched
    assert engine.calls == 1
