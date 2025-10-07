"""Chunking utilities used by the ingestion pipeline."""

from __future__ import annotations

import re
from typing import List, Sequence

try:  # Optional dependency used for semantic chunking
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - triggered only when dependency missing
    SentenceTransformer = None  # type: ignore

DEFAULT_SEPARATORS: Sequence[str] = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ")


def chunk_text(
    normalised_text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    strategy: str = "recursive",
    semantic_model_name: str | None = None,
) -> List[str]:
    """Split text into overlapping chunks.

    Parameters
    ----------
    normalised_text:
        Text to split.  The input is expected to be pre-processed by the
        normaliser to remove spurious whitespace.
    chunk_size:
        Approximate maximum chunk size in characters.
    chunk_overlap:
        Size of the character overlap between consecutive chunks.
    strategy:
        Either ``"recursive"`` for the standard hierarchical splitter or
        ``"semantic"`` which leverages sentence embeddings to place boundaries.
    semantic_model_name:
        When strategy is ``"semantic"`` this optional argument controls which
        sentence-transformers model should be loaded.
    """

    text = normalised_text.strip()
    if not text:
        return []

    if strategy == "semantic" and SentenceTransformer is not None:
        model_name = semantic_model_name or "intfloat/multilingual-e5-base"
        model = SentenceTransformer(model_name)
        return _semantic_chunk(text, model, chunk_size)

    return _recursive_chunk(text, chunk_size, chunk_overlap)


def _recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    for separator in DEFAULT_SEPARATORS:
        parts = text.split(separator)
        if len(parts) == 1:
            continue
        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = separator.join(filter(None, [current, part])).strip()
            if not candidate:
                continue
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(part) <= chunk_size:
                current = part
            else:
                chunks.extend(_recursive_chunk(part, chunk_size, chunk_overlap))
                current = ""
        if current:
            chunks.append(current)
        break
    else:  # fallback when no separator produced chunks
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    return _apply_overlap(chunks, chunk_overlap)


def _apply_overlap(chunks: Sequence[str], overlap: int) -> List[str]:
    if not chunks:
        return []
    if overlap <= 0:
        return list(chunks)

    window: List[str] = []
    for chunk in chunks:
        if not window:
            window.append(chunk)
            continue
        last = window[-1]
        tail = last[-overlap:]
        window.append((tail + " " + chunk).strip())
    return window


def _semantic_chunk(text: str, model: SentenceTransformer, chunk_size: int) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences, batch_size=32, normalize_embeddings=True)
    boundaries = [0]
    for idx in range(len(sentences) - 1):
        similarity = float(embeddings[idx] @ embeddings[idx + 1])
        if similarity < 0.55:
            boundaries.append(idx + 1)
    boundaries.append(len(sentences))

    chunks: List[str] = []
    i = 0
    while i < len(boundaries) - 1:
        start = boundaries[i]
        end = boundaries[i + 1]
        candidate = " ".join(sentences[start:end]).strip()
        while end < len(sentences) and len(candidate) < chunk_size:
            end += 1
            candidate = " ".join(sentences[start:end]).strip()
        chunks.append(candidate)
        i += 1
    return chunks


def _split_sentences(text: str) -> List[str]:
    splitter = re.compile(r"(?<=[\.!?])\s+(?=[A-ZА-ЯЁ])")
    sentences = [sentence.strip() for sentence in splitter.split(text) if sentence.strip()]
    if not sentences:
        return [text]
    return sentences


__all__ = ["chunk_text"]
