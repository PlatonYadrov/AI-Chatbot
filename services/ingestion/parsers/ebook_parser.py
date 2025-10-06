"""
eBook парсер: EPUB/FB2 извлечение текста и структуры.
Вход: raw_docs; Выход: parsed_docs.
"""

from __future__ import annotations

from typing import Any, Dict


def parse_document(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = raw_doc.get("doc_id")
    text = ""  # TODO: epub/FB2 libs
    structure = {"chapters": []}
    source_meta = {"source": raw_doc.get("source"), **(raw_doc.get("meta") or {})}
    return {"doc_id": doc_id, "text": text, "structure": structure, "source_meta": source_meta}


__all__ = ["parse_document"]

