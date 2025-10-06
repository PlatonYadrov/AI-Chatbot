"""
Pydantic-схемы общего пользования.

Содержит:
- DocumentRaw, ParsedDoc, ChunkReady, EmbeddingPoint
- SearchFilters, SearchResult, ChatAnswer
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocumentRaw(BaseModel):
    doc_id: str
    source: str
    url: Optional[str] = None
    path: Optional[str] = None
    mime: Optional[str] = None
    acl: List[str] = []
    meta: Dict[str, Any] = {}


class ParsedDoc(BaseModel):
    doc_id: str
    text: str
    structure: Dict[str, Any]
    source_meta: Dict[str, Any]


class ChunkReady(BaseModel):
    doc_id: str
    chunk_idx: int
    text: str
    payload: Dict[str, Any]


class EmbeddingPoint(BaseModel):
    doc_id: str
    chunk_idx: int
    vector: List[float]
    payload: Dict[str, Any]


class SearchFilters(BaseModel):
    acl: List[str] = []
    manager_type: List[str] = []
    category: List[str] = []
    effective: str = "now"


class SearchResult(BaseModel):
    point_id: str
    score: float
    payload: Dict[str, Any]


class ChatAnswer(BaseModel):
    text: str
    usage: Dict[str, Any]
    citations: List[Dict[str, Any]] = []


__all__ = [
    "DocumentRaw",
    "ParsedDoc",
    "ChunkReady",
    "EmbeddingPoint",
    "SearchFilters",
    "SearchResult",
    "ChatAnswer",
]

