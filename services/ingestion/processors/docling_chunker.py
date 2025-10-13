"""Advanced chunking using Docling's HybridChunker.

Docling's HybridChunker uses document structure and semantic understanding 
to determine optimal chunk boundaries. It considers:
- Document layout (headings, paragraphs, sections)
- Semantic coherence
- Token limits
- Overlap strategy

This is superior to naive text splitting as it preserves document structure.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

# ВАЖНО: импорт отделяем от конструирования
try:
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
    DOCLING_CHUNKING_AVAILABLE = True
    LOGGER.info("docling_chunking_import_ok")
except Exception as e:
    DOCLING_CHUNKING_AVAILABLE = False
    LOGGER.error("docling_chunker_import_failed", exc_info=True)


@dataclass
class DoclingChunk:
    """Chunk produced by Docling's HybridChunker."""
    
    text: str
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"text": self.text, "metadata": self.meta}


class DoclingChunker:
    """Wrapper for Docling's HybridChunker with smart defaults.
    
    Uses HybridChunker which combines:
    - Hierarchical chunking (respects document structure)
    - Token-aware splitting
    - Semantic boundary detection
    """
    
    def __init__(
        self,
        *,
        tokenizer: str = "intfloat/multilingual-e5-large",  # по умолчанию HF-эмбеддер
        max_tokens: int = 512,
        overlap: int = 0,               # HybridChunker обычно работает без оверлапа
        merge_peers: bool = True,
        include_metadata: bool = True,
    ) -> None:
        """Initialize Docling chunker.
        
        Args:
            tokenizer: HuggingFace model ID for tokenizer
                       - "intfloat/multilingual-e5-large" (default, matches embeddings)
                       - "bert-base-uncased" for BERT models
                       - любой HuggingFace model ID
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks (manual overlap, HybridChunker doesn't use it)
            merge_peers: Merge small adjacent chunks for better context
            include_metadata: Include document structure metadata in chunks
        """
        self.tokenizer_name = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.merge_peers = merge_peers
        self.include_metadata = include_metadata
        
        # Lazy initialization - only import when needed
        self._chunker = None
    
    def _build_tokenizer(self):
        """
        Создаёт объект HuggingFace токенизатора для HybridChunker.
        Загружает токенизатор из HuggingFace Hub по model ID.
        """
        hf_tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=self.max_tokens)

    def _get_chunker(self):
        """Lazy load HybridChunker."""
        if self._chunker is None:
            if not DOCLING_CHUNKING_AVAILABLE:
                raise RuntimeError("Docling chunker unavailable (import failed earlier).")

            try:
                tokenizer_obj = self._build_tokenizer()
                self._chunker = HybridChunker(
                    tokenizer=tokenizer_obj,
                    merge_peers=self.merge_peers,
                )
                LOGGER.info(
                    "docling_chunker_initialized",
                    extra={
                        "tokenizer": self.tokenizer_name,
                        "max_tokens": self.max_tokens,
                        "merge_peers": self.merge_peers,
                        "overlap": self.overlap,
                    },
                )
            except Exception:
                LOGGER.error("docling_chunker_init_failed", exc_info=True)
                raise
        
        return self._chunker
    
    def chunk_document(
        self,
        docling_doc,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DoclingChunk]:
        """Chunk a DoclingDocument using HybridChunker.
        
        Args:
            docling_doc: DoclingDocument object from Docling parser
            doc_metadata: Additional metadata to include in chunks
            
        Returns:
            List of DoclingChunk objects with text and metadata
        """
        chunker = self._get_chunker()
        doc_metadata = doc_metadata or {}
        
        chunks: List[DoclingChunk] = []
        try:
            for i, ch in enumerate(chunker.chunk(dl_doc=docling_doc)):
                text = getattr(ch, "text", str(ch)).strip()
                if not text:
                    continue

                # обогащённый текст для эмбеддинга (с заголовками/подписями)
                try:
                    enriched = chunker.contextualize(ch)
                except Exception:
                    enriched = text

                meta = {
                    **doc_metadata,
                    "chunk_index": i,
                    "chunking_strategy": "docling_hybrid",
                    "enriched_text": enriched,
                }

                # переносим полезные поля, если есть
                for k in ("headings", "caption", "element_type", "page_no", "pages"):
                    if hasattr(ch, k):
                        meta[k] = getattr(ch, k)

                chunks.append(DoclingChunk(text=text, meta=meta))

            LOGGER.info("docling_chunking_complete", extra={"chunks": len(chunks)})
            return chunks

        except Exception:
            LOGGER.exception("docling_chunking_failed")
            raise
    
    def chunk_document_lazy(
        self,
        docling_doc,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[DoclingChunk]:
        """Lazy version that yields chunks one by one (memory efficient).
        
        Args:
            docling_doc: DoclingDocument object from Docling parser
            doc_metadata: Additional metadata to include in chunks
            
        Yields:
            DoclingChunk objects with text and metadata
        """
        chunker = self._get_chunker()
        doc_metadata = doc_metadata or {}
        try:
            for i, ch in enumerate(chunker.chunk(dl_doc=docling_doc)):
                text = getattr(ch, "text", str(ch)).strip()
                if not text:
                    continue
                try:
                    enriched = chunker.contextualize(ch)
                except Exception:
                    enriched = text

                meta = {
                    **doc_metadata,
                    "chunk_index": i,
                    "chunking_strategy": "docling_hybrid",
                    "enriched_text": enriched,
                }
                for k in ("headings", "caption", "element_type", "page_no", "pages"):
                    if hasattr(ch, k):
                        meta[k] = getattr(ch, k)

                yield DoclingChunk(text=text, meta=meta)
        except Exception:
            LOGGER.exception("docling_chunking_failed")
            raise


def chunk_docling_document(
    docling_doc,
    doc_metadata: Optional[Dict[str, Any]] = None,
    max_tokens: int = 512,
    overlap: int = 0,
    tokenizer: str = "intfloat/multilingual-e5-large",
) -> List[DoclingChunk]:
    """Convenience function to chunk a Docling document.
    
    Args:
        docling_doc: DoclingDocument object from Docling parser
        doc_metadata: Additional metadata to include in chunks
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        tokenizer: HuggingFace model ID for tokenizer
        
    Returns:
        List of DoclingChunk objects
    """
    return DoclingChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=overlap,
    ).chunk_document(docling_doc, doc_metadata)


__all__ = ["DoclingChunker", "DoclingChunk", "chunk_docling_document"]

