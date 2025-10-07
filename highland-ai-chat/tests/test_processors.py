"""Unit tests for ingestion processors."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ingestion.processors.chunker import chunk_text
from services.ingestion.processors.deduplicator import Deduplicator
from services.ingestion.processors.metadata_extractor import enrich_metadata
from services.ingestion.processors.normalizer import normalise_text
from services.ingestion.processors.pii_redactor import redact_pii
from services.ingestion.parsers.base import RawBlock


def test_normalise_text_removes_control_characters() -> None:
    raw = "Hello\u0007 World\nPage 1"
    assert normalise_text(raw) == "Hello World"


def test_chunk_text_recursive_strategy_produces_overlap() -> None:
    text = "Paragraph one. Paragraph two. Paragraph three." * 5
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)
    assert len(chunks) >= 2
    assert chunks[0][-10:] in chunks[1]


def test_deduplicator_filters_identical_blocks() -> None:
    blocks = [
        RawBlock(text="Same", meta={}),
        RawBlock(text="Same", meta={}),
        RawBlock(text="Different", meta={}),
    ]
    filtered = list(Deduplicator().filter_blocks(blocks))
    assert [block.text for block in filtered] == ["Same", "Different"]


def test_metadata_enrichment_adds_version_and_dates() -> None:
    meta = enrich_metadata({}, "Version v1.2 released 01.02.2024")
    assert meta["version"] == "1.2"
    assert meta["effective_from"] == "2024-02-01"


def test_redact_pii_masks_sensitive_patterns() -> None:
    text = "Call me at +1 555-123-4567 or email test@example.com"
    redacted = redact_pii(text)
    assert "test@example.com" not in redacted
    assert "+1 555-123-4567" not in redacted
