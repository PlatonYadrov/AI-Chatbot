"""Qdrant indexer service implementation."""

from __future__ import annotations

from typing import Iterable, Mapping

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - triggered when dependency missing
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


class QdrantIndexer:
    """Lightweight wrapper around :mod:`qdrant-client` for batch upserts."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        vector_size: int = 1024,
    ) -> None:
        if QdrantClient is None:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "qdrant-client is required for indexing. Install it with `pip install qdrant-client`."
            )
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        assert qmodels is not None
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=self.vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert_embeddings(self, points: Iterable[Mapping[str, object]]) -> None:
        assert qmodels is not None
        payloads = []
        vectors = []
        ids = []
        for point in points:
            ids.append(point.get("id"))
            vectors.append(point["vector"])  # type: ignore[index]
            payloads.append(point.get("payload", {}))
        self.client.upsert(
            collection_name=self.collection,
            points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )


__all__ = ["QdrantIndexer"]
