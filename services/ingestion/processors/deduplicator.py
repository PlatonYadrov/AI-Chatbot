"""Document and chunk deduplication utilities."""

from __future__ import annotations

import hashlib
import logging
from typing import List, Set, Dict, Any

try:
    from datasketch import MinHash, MinHashLSH
    _MINHASH_OK = True
except ImportError:
    _MINHASH_OK = False


def _shingles(text: str, k: int = 5) -> Set[str]:
    """Generate character k-shingles."""
    text_clean = "".join(text.lower().split())
    return {text_clean[i:i+k] for i in range(len(text_clean) - k + 1)}


def _minhash_signature(text: str, num_perm: int = 128) -> MinHash:
    """Generate MinHash signature."""
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


LOGGER = logging.getLogger(__name__)


def deduplicate_chunks(chunks: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Deduplicate chunks using MinHash LSH.
    
    Args:
        chunks: list of chunk dicts with 'text' and 'metadata'
        threshold: Jaccard similarity threshold (0.8 = 80%)
    
    Returns:
        Deduplicated list of chunks
    """
    if not _MINHASH_OK:
        LOGGER.info("dedup_fallback_exact")
        return _exact_hash_dedup(chunks)
    
    # Lower threshold or increase num_perm to avoid "bands too small" error
    # threshold=0.99 is too high for num_perm=128
    effective_threshold = min(threshold, 0.95)  # Cap at 0.95
    lsh = MinHashLSH(threshold=effective_threshold, num_perm=128)
    unique_chunks: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text.strip():
            continue
        
        chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{idx}")
        # Skip if the key was already inserted earlier in this run
        if chunk_id in seen_ids:
            continue
        mh = _minhash_signature(text)
        
        # Query LSH for duplicates
        try:
            duplicates = lsh.query(mh)
        except Exception:
            LOGGER.debug("lsh_query_failed")
            duplicates = []
        if duplicates:
            # Skip if similar chunk already indexed
            continue
        
        # Insert and keep
        try:
            lsh.insert(chunk_id, mh)
        except Exception:
            # Key may already exist (e.g., repeated chunk_id); skip silently
            LOGGER.debug("lsh_insert_failed", extra={"chunk_id": chunk_id})
            continue
        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)
    
    LOGGER.info("dedup_result", extra={"input": len(chunks), "unique": len(unique_chunks)})
    return unique_chunks


def _exact_hash_dedup(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback: exact content hash deduplication."""
    seen_hashes: Set[str] = set()
    unique_chunks: List[Dict[str, Any]] = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text.strip():
            continue
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_chunks.append(chunk)
    
    LOGGER.info("exact_dedup_result", extra={"input": len(chunks), "unique": len(unique_chunks)})
    return unique_chunks


def deduplicate_documents(documents: List[str]) -> List[str]:
    """
    Deduplicate full documents by content hash.
    
    Args:
        documents: list of document texts
    
    Returns:
        Deduplicated list of documents
    """
    seen: Set[str] = set()
    unique: List[str] = []
    for doc in documents:
        h = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


__all__ = ["deduplicate_chunks", "deduplicate_documents"]
