"""End-to-end orchestration helpers for the ingestion pipeline.

The documentation already describes how raw documents should flow through
parsers, processors, embedder and indexer components.  This module brings the
pieces together so that engineers can experiment with the pipeline without
having to wire everything manually in notebooks or ad-hoc scripts.

The primary entry point is :class:`LocalIngestionPipeline` which accepts a
mapping of file extensions to parser instances, a batch embedder implementation
and an optional indexer.  It is intentionally lightweight yet fully functional:

* normalises raw blocks emitted by parsers,
* performs overlap-aware chunking,
* redacts common PII patterns,
* enriches the metadata payload with basic heuristics,
* deduplicates identical chunks across the entire run,
* produces embedding vectors and (optionally) upserts them into Qdrant.

The resulting chunk records can then be fed into tests or further processing
steps such as hybrid retrieval evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from .parsers import BaseParser, RawBlock
from .processors.chunker import chunk_text
from .processors.deduplicator import Deduplicator
from .processors.metadata_extractor import enrich_metadata
from .processors.normalizer import normalise_text
from .processors.pii_redactor import redact_pii


@dataclass(slots=True)
class ChunkRecord:
    """Final artefact produced by the ingestion pipeline."""

    id: str
    text: str
    vector: List[float]
    payload: Dict[str, object]


class LocalIngestionPipeline:
    """Utility class to ingest local files using the existing building blocks."""

    def __init__(
        self,
        *,
        parsers: Mapping[str, BaseParser],
        embedder,
        indexer: Optional[object] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        chunk_strategy: str = "recursive",
        deduplicator: Optional[Deduplicator] = None,
    ) -> None:
        if not parsers:
            raise ValueError("At least one parser must be provided")
        self.parsers: Dict[str, BaseParser] = {
            ext.lower().lstrip("."): parser for ext, parser in parsers.items()
        }
        self.embedder = embedder
        self.indexer = indexer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.deduplicator = deduplicator or Deduplicator()

    def ingest_path(self, path: str | Path, *, doc_id: Optional[str] = None) -> List[ChunkRecord]:
        """Ingest a single file returning the generated chunk records."""

        path = Path(path)
        parser = self._resolve_parser(path)
        if parser is None:
            return []

        prepared_blocks: List[RawBlock] = []
        chunk_index = 0

        for raw_block in parser.parse(str(path), doc_id=doc_id):
            normalised = normalise_text(raw_block.text)
            if not normalised:
                continue
            chunks = chunk_text(
                normalised,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=self.chunk_strategy,
            )
            for chunk in chunks:
                redacted = redact_pii(chunk)
                meta: MutableMapping[str, object] = dict(raw_block.meta)
                meta.setdefault("doc_id", doc_id or meta.get("doc_id") or path.stem)
                meta.setdefault("path", str(path))
                meta["chunk_index"] = chunk_index
                chunk_index += 1
                enriched = enrich_metadata(meta, redacted)
                prepared_blocks.append(RawBlock(text=redacted, meta=enriched))

        unique_blocks = list(self.deduplicator.filter_blocks(prepared_blocks))
        if not unique_blocks:
            return []

        vectors = self.embedder.embed_passages([block.text for block in unique_blocks])
        records = []
        points = []
        for block, vector in zip(unique_blocks, vectors):
            vector_list = _vector_to_list(vector)
            payload = dict(block.meta)
            payload.setdefault("text", block.text)
            point_id = _make_point_id(payload, block.text)
            record = ChunkRecord(id=point_id, text=block.text, vector=vector_list, payload=payload)
            records.append(record)
            points.append({"id": point_id, "vector": vector_list, "payload": payload})

        if self.indexer is not None:
            upsert = getattr(self.indexer, "upsert_embeddings", None)
            if callable(upsert):
                upsert(points)

        return records

    def ingest_directory(self, root: str | Path) -> List[ChunkRecord]:
        """Recursively ingest all supported files within *root*."""

        root_path = Path(root)
        results: List[ChunkRecord] = []
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue
            results.extend(self.ingest_path(file_path))
        return results

    def _resolve_parser(self, path: Path) -> Optional[BaseParser]:
        return self.parsers.get(path.suffix.lower().lstrip("."))


def _vector_to_list(vector: Iterable[float] | object) -> List[float]:
    """Convert vectors returned by various libraries into a plain list."""

    if isinstance(vector, list):
        return [float(value) for value in vector]
    if hasattr(vector, "tolist"):
        return [float(value) for value in vector.tolist()]
    return [float(value) for value in vector]  # type: ignore[arg-type]


def _make_point_id(meta: Mapping[str, object], text: str) -> str:
    base = str(meta.get("doc_id") or meta.get("path") or "unknown")
    chunk_idx = meta.get("chunk_index", "0")
    fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest()
    raw = json.dumps({"base": base, "chunk": chunk_idx, "fp": fingerprint}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["ChunkRecord", "LocalIngestionPipeline"]

