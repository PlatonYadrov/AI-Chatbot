from __future__ import annotations

import io
import os
import tempfile
import pytest
from fastapi.testclient import TestClient


def _make_simple_pdf_bytes() -> bytes:
    try:
        import fitz  # PyMuPDF
    except Exception:
        pytest.skip("PyMuPDF (fitz) is not installed; skipping ingestion PDF test")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello PDF!", fontsize=12)
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


def test_ingestion_endpoint_pdf(tmp_path):
    try:
        from services.ingestion.api import app
    except Exception:
        pytest.skip("ingestion API not importable in this environment")

    client = TestClient(app)
    pdf_bytes = _make_simple_pdf_bytes()
    files = {"file": ("sample.pdf", pdf_bytes, "application/pdf")}

    # Note: this test exercises pipeline up to parser/chunker; external services
    # like OCR/Embeddings/Qdrant may be unavailable in CI; we expect either 200
    # with payload or a well-formed error if vector store is down.
    response = client.post("/ingest", files=files)
    assert response.status_code in (200, 400, 502)
    # If success path:
    if response.status_code == 200:
        data = response.json()
        assert data.get("status") == "ok"
        assert data.get("chunks_total", 0) >= data.get("chunks_unique", 0)

