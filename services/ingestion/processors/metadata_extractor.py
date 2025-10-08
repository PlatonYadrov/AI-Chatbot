"""Metadata extraction utilities."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Dict, Any, Optional


def extract_metadata(
    chunk_text: str,
    doc_metadata: Optional[Dict[str, Any]] = None,
    chunk_index: int = 0,
) -> Dict[str, Any]:
    """
    Extract and enrich chunk metadata.
    
    Args:
        chunk_text: chunk text
        doc_metadata: document-level metadata
        chunk_index: chunk index in document
    
    Returns:
        Enriched metadata dict
    """
    doc_metadata = doc_metadata or {}
    
    metadata = {
        **doc_metadata,
        "chunk_index": chunk_index,
        "chunk_id": f"{doc_metadata.get('doc_id', 'unknown')}_{chunk_index}",
        "chunk_hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16],
        "char_count": len(chunk_text),
        "indexed_at": datetime.utcnow().isoformat(),
    }
    
    # Optional: detect language (requires langdetect)
    try:
        from langdetect import detect
        metadata["language"] = detect(chunk_text)
    except Exception:
        metadata["language"] = doc_metadata.get("language", "unknown")
    
    return metadata


__all__ = ["extract_metadata"]
