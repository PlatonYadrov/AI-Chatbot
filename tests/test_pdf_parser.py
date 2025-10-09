from __future__ import annotations

import os
import tempfile
import pytest


def _make_simple_pdf(path: str) -> None:
    try:
        import fitz  # PyMuPDF
    except Exception:
        pytest.skip("PyMuPDF (fitz) is not installed; skipping PDF parser test")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF!", fontsize=12)
    doc.save(path)
    doc.close()


def test_pdf_parser_basic_blocks():
    try:
        from services.ingestion.parsers.pdf_parser import PdfParser
    except Exception:
        pytest.skip("ingestion module not importable in this environment")

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "sample.pdf")
        _make_simple_pdf(pdf_path)

        parser = PdfParser(extract_tables=False)
        blocks = list(parser.parse(pdf_path))

        assert blocks, "Parser should produce at least one block"
        for b in blocks:
            assert b.text.strip(), "Block text should not be empty"
            assert b.meta.get("type") == "pdf"
            assert b.meta.get("path") == pdf_path

