"""Batch embedding service implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - triggered when dependency missing
    SentenceTransformer = None  # type: ignore


class EmbeddingCache:
    """Persistent on-disk cache keyed by SHA1 of the text payload."""

    def __init__(self, path: str | Path = "embeddings.cache.json") -> None:
        self.path = Path(path)
        if self.path.exists():
            self._store = json.loads(self.path.read_text("utf-8"))
        else:
            self._store: dict[str, list[float]] = {}

    def get(self, text: str) -> list[float] | None:
        return self._store.get(self._key(text))

    def set(self, text: str, vector: Sequence[float]) -> None:
        self._store[self._key(text)] = list(vector)
        self.path.write_text(json.dumps(self._store), encoding="utf-8")

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()


class BatchEmbedder:
    """Compute embeddings for passages using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        *,
        device: str = "cpu",
        cache: EmbeddingCache | None = None,
    ) -> None:
        if SentenceTransformer is None:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "sentence-transformers is required for the embedding service. "
                "Install it with `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.cache = cache or EmbeddingCache()

    def embed_passages(self, texts: Iterable[str]) -> List[List[float]]:
        prepared: List[List[float] | str] = []
        uncached: list[str] = []
        for text in texts:
            cached = self.cache.get(text)
            if cached is not None:
                prepared.append(cached)
            else:
                prepared.append(text)
                uncached.append(text)

        if uncached:
            payload = self._add_prefix(uncached)
            vectors = self.model.encode(payload, batch_size=32, normalize_embeddings=True)
            iterator = iter(vectors)
            for idx, item in enumerate(prepared):
                if isinstance(item, list):
                    continue
                vector = next(iterator)
                vector_list = vector.tolist()
                self.cache.set(item, vector_list)
                prepared[idx] = vector_list
        return [item if isinstance(item, list) else self.cache.get(item) or [] for item in prepared]

    def embed_query(self, query: str) -> List[float]:
        payload = self._add_prefix([query], prefix="query: ")[0]
        vector = self.model.encode([payload], normalize_embeddings=True)[0]
        return vector.tolist()

    def _add_prefix(self, texts: Iterable[str], *, prefix: str = "passage: ") -> List[str]:
        if "e5" in self.model_name:
            return [f"{prefix}{text}" for text in texts]
        return list(texts)


__all__ = ["BatchEmbedder", "EmbeddingCache"]
