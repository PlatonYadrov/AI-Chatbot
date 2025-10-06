"""
Назначение: извлечь текст/структуру/таблицы из PDF.
Вход: raw_docs (dict с ключами: doc_id, source, url|path, mime, acl, meta)
Выход: parsed_docs (dict: {doc_id, text, structure, source_meta})
SLO: сред. < 3 s/док; OCR — отдельная очередь raw_docs_ocr.
"""

from __future__ import annotations

from typing import Any, Dict


def parse_document(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Минимальный каркас парсера PDF. Реализация Tika/Tesseract — позже.

    Возвращает структуру parsed_docs с заглушками.
    """
    doc_id = raw_doc.get("doc_id")
    text = ""  # TODO: внедрить извлечение текста из PDF
    structure = {"title": None, "headings": [], "tables": []}
    source_meta = {"source": raw_doc.get("source"), **(raw_doc.get("meta") or {})}
    return {"doc_id": doc_id, "text": text, "structure": structure, "source_meta": source_meta}


__all__ = ["parse_document"]

