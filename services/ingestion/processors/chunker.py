"""
Чанкование: size=512, overlap=64, по границам предложений (упрощённо).
Вход: unique_docs; Выход: ready_for_embedding (chunks).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _sentences(text: str) -> List[str]:
    # Упрощённый сплит по точке/вопросу/восклицанию
    import re

    parts = re.split(r"(?<=[\.!?])\s+", text or "")
    return [p.strip() for p in parts if p.strip()]


def chunkify(doc: Dict[str, Any], size: int = 512, overlap: int = 64) -> Iterable[Dict[str, Any]]:
    sentences = _sentences(doc.get("text", ""))
    buffer: List[str] = []
    current_len = 0
    chunk_idx = 0
    for sent in sentences:
        if current_len + len(sent) + 1 > size and buffer:
            chunk_text = " ".join(buffer)
            yield {
                "doc_id": doc.get("doc_id"),
                "chunk_idx": chunk_idx,
                "text": chunk_text,
                "payload": doc.get("source_meta", {}),
            }
            chunk_idx += 1
            # overlap
            tail = chunk_text[-overlap:]
            buffer = [tail]
            current_len = len(tail)
        buffer.append(sent)
        current_len += len(sent) + 1
    if buffer:
        chunk_text = " ".join(buffer)
        yield {
            "doc_id": doc.get("doc_id"),
            "chunk_idx": chunk_idx,
            "text": chunk_text,
            "payload": doc.get("source_meta", {}),
        }


__all__ = ["chunkify"]

