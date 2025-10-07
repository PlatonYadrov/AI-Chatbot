"""Tests for the end-to-end ingestion pipeline helper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.ingestion.parsers.base import BaseParser, RawBlock
from services.ingestion.pipeline import ChunkRecord, LocalIngestionPipeline


class DummyParser(BaseParser):
    """Minimal parser used to exercise the pipeline without external deps."""

    def __init__(self, blocks: Optional[List[str]] = None) -> None:
        self.blocks = blocks or [
            "Напишите нам: sales@example.com",
            "Регламент KPI утверждён 01.01.2024",
            "Регламент KPI утверждён 01.01.2024",
        ]

    def parse(self, path: str | Path, *, doc_id: Optional[str] = None) -> Iterator[RawBlock]:
        for text in self.blocks:
            yield RawBlock(text=text, meta={"path": str(path), "doc_id": doc_id})


class DummyEmbedder:
    def embed_passages(self, texts: Iterable[str]) -> List[List[float]]:
        return [[float(len(text))] for text in texts]


class RecordingIndexer:
    def __init__(self) -> None:
        self.points: List[Mapping[str, object]] = []

    def upsert_embeddings(self, points: Iterable[Mapping[str, object]]) -> None:
        self.points.extend(points)


def test_ingest_path_produces_records_and_indexes(tmp_path: Path) -> None:
    indexer = RecordingIndexer()
    pipeline = LocalIngestionPipeline(
        parsers={"txt": DummyParser()},
        embedder=DummyEmbedder(),
        indexer=indexer,
        chunk_size=64,
        chunk_overlap=0,
    )

    doc_path = tmp_path / "document.txt"
    doc_path.write_text("placeholder", encoding="utf-8")

    records = pipeline.ingest_path(doc_path, doc_id="doc-001")

    assert len(records) == 2  # duplicate block removed
    assert all(isinstance(record, ChunkRecord) for record in records)

    first_payload = records[0].payload
    assert first_payload["doc_id"] == "doc-001"
    assert first_payload["chunk_index"] == 0
    assert "[EMAIL]" in records[0].text  # PII redaction applied
    assert "effective_from" in records[1].payload  # metadata enrichment picked up date

    assert indexer.points  # indexer received upsert payloads
    assert indexer.points[0]["payload"]["text"] == records[0].text
    assert records[0].vector == [float(len(records[0].text))]


def test_ingest_directory_skips_unsupported_extensions(tmp_path: Path) -> None:
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("content", encoding="utf-8")
    doc2 = tmp_path / "doc2.bin"
    doc2.write_text("ignored", encoding="utf-8")

    pipeline = LocalIngestionPipeline(
        parsers={"txt": DummyParser(blocks=["A", "B"])},
        embedder=DummyEmbedder(),
        chunk_size=32,
        chunk_overlap=0,
    )

    records = pipeline.ingest_directory(tmp_path)

    assert len(records) == 2
    assert {record.payload["path"] for record in records} == {str(doc1)}

