"""Document and chunk deduplication utilities."""

from __future__ import annotations

import hashlib
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
        # Fallback: exact hash dedup
        return _exact_hash_dedup(chunks)
    
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_chunks: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text.strip():
            continue
        
        chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{idx}")
        mh = _minhash_signature(text)
        
        # Query LSH for duplicates
        duplicates = lsh.query(mh)
        if duplicates:
            # Skip if similar chunk already indexed
            continue
        
        # Insert and keep
        lsh.insert(chunk_id, mh)
        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)
    
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
