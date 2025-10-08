"""Production chunking with token-aware, sentence boundaries, and metadata."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False


DEFAULT_SEPARATORS: Sequence[str] = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ")


class Chunk:
    """Represents a text chunk with metadata."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata}


def _load_config() -> dict:
    config_path = Path(__file__).parents[3] / "configs" / "model_config.yaml"
    if _YAML_OK and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_tokenizer(model_name: str = "cl100k_base"):
    if _TIKTOKEN_OK:
        try:
            return tiktoken.get_encoding(model_name)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    return None


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer:
        return len(tokenizer.encode(text))
    # Fallback: rough estimate (1 token ≈ 4 chars)
    return len(text) // 4


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex-based)."""
    splitter = re.compile(r"(?<=[\.!?])\s+(?=[A-ZА-ЯЁ])")
    sentences = [s.strip() for s in splitter.split(text) if s.strip()]
    return sentences if sentences else [text]


def chunk_text(
    normalised_text: str,
    doc_metadata: Optional[Dict[str, Any]] = None,
    chunk_size_tokens: int = 300,
    overlap_tokens: int = 75,
    min_chunk_tokens: int = 50,
    max_chunk_tokens: int = 512,
    lang: str = "en",
    strategy: str = "token_aware",
) -> List[Chunk]:
    """
    Chunk text with token-aware sentence boundaries and metadata.
    
    Args:
        normalised_text: cleaned text
        doc_metadata: document metadata (doc_id, source_uri, etc.)
        chunk_size_tokens: target chunk size in tokens
        overlap_tokens: overlap in tokens
        min_chunk_tokens: minimum chunk size (discard smaller)
        max_chunk_tokens: hard limit
        lang: language code
        strategy: 'token_aware', 'recursive', or 'semantic'
    
    Returns:
        List of Chunk objects with text and metadata
    """
    config = _load_config().get("chunking", {})
    chunk_size_tokens = config.get("chunk_size_tokens", chunk_size_tokens)
    overlap_tokens = config.get("overlap_tokens", overlap_tokens)
    min_chunk_tokens = config.get("min_chunk_tokens", min_chunk_tokens)
    max_chunk_tokens = config.get("max_chunk_tokens", max_chunk_tokens)
    strategy = config.get("strategy", strategy)
    
    tokenizer = _get_tokenizer()
    doc_metadata = doc_metadata or {}
    
    text = normalised_text.strip()
    if not text:
        return []
    
    # Choose strategy
    if strategy == "semantic" and _ST_OK:
        raw_chunks = _semantic_chunk(text, chunk_size_tokens * 4)  # approx chars
    elif strategy == "token_aware":
        raw_chunks = _token_aware_chunk(text, tokenizer, chunk_size_tokens, overlap_tokens, max_chunk_tokens)
    else:
        # Fallback to recursive char-based
        raw_chunks = _recursive_chunk(text, chunk_size_tokens * 4, overlap_tokens * 4)
    
    # Build Chunk objects with metadata
    chunks: List[Chunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        tokens = _count_tokens(chunk_text, tokenizer)
        if tokens < min_chunk_tokens:
            continue
        chunk_metadata = {
            **doc_metadata,
            "chunk_index": idx,
            "chunk_id": f"{doc_metadata.get('doc_id', 'unknown')}_{idx}",
            "chunk_hash": _hash_text(chunk_text),
            "tokens": tokens,
            "language": lang,
        }
        chunks.append(Chunk(chunk_text, chunk_metadata))
    
    return chunks


def _token_aware_chunk(
    text: str,
    tokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_chunk_tokens: int,
) -> List[str]:
    """Token-aware sentence-boundary chunking."""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    
    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = _count_tokens(sent, tokenizer)
        
        if current_tokens + sent_tokens > chunk_size_tokens and current_sentences:
            # Finalize current chunk
            chunks.append(" ".join(current_sentences))
            
            # Keep overlap sentences
            overlap_sents = []
            overlap_tok = 0
            for s in reversed(current_sentences):
                s_tok = _count_tokens(s, tokenizer)
                if overlap_tok + s_tok <= overlap_tokens:
                    overlap_sents.insert(0, s)
                    overlap_tok += s_tok
                else:
                    break
            current_sentences = overlap_sents
            current_tokens = overlap_tok
        
        current_sentences.append(sent)
        current_tokens += sent_tokens
        
        # Hard limit
        if current_tokens > max_chunk_tokens:
            chunks.append(" ".join(current_sentences))
            current_sentences = []
            current_tokens = 0
    
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    
    return chunks


def _recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursive char-based chunking (fallback)."""
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
    else:
        step = max(1, chunk_size - chunk_overlap)
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]
    
    return _apply_overlap(chunks, chunk_overlap)


def _apply_overlap(chunks: Sequence[str], overlap: int) -> List[str]:
    if not chunks or overlap <= 0:
        return list(chunks)
    window: List[str] = []
    for chunk in chunks:
        if not window:
            window.append(chunk)
        else:
            tail = window[-1][-overlap:]
            window.append((tail + " " + chunk).strip())
    return window


def _semantic_chunk(text: str, chunk_size: int) -> List[str]:
    """Semantic chunking using sentence embeddings."""
    if not _ST_OK:
        return _recursive_chunk(text, chunk_size, chunk_size // 5)
    
    model = SentenceTransformer("intfloat/multilingual-e5-base")
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


__all__ = ["chunk_text", "Chunk"]
