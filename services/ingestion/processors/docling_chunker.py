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
        tokenizer: str = "bert-base-uncased",
        max_tokens: int = 512,
        overlap: int = 75,
        merge_peers: bool = True,
        include_metadata: bool = True,
    ) -> None:
        """Initialize Docling chunker.
        
        Args:
            tokenizer: Tokenizer name for HybridChunker
                       - "bert-base-uncased" for BERT/e5/BGE models (default)
                       - "cl100k_base" for GPT-3.5/4
                       - "gpt2" for GPT-2
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks
            merge_peers: Merge small adjacent chunks for better context
            include_metadata: Include document structure metadata in chunks
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.merge_peers = merge_peers
        self.include_metadata = include_metadata
        
        # Lazy initialization - only import when needed
        self._chunker = None
    
    def _get_chunker(self):
        """Lazy load HybridChunker."""
        if self._chunker is None:
            try:
                from docling_core.transforms.chunker import HybridChunker
                
                self._chunker = HybridChunker(
                    tokenizer=self.tokenizer,
                    max_tokens=self.max_tokens,
                    merge_peers=self.merge_peers,
                )
                LOGGER.info(
                    "docling_chunker_initialized",
                    extra={
                        "tokenizer": self.tokenizer,
                        "max_tokens": self.max_tokens,
                        "overlap": self.overlap,
                    },
                )
            except ImportError as e:
                LOGGER.error(
                    "docling_chunker_import_failed",
                    extra={"error": str(e)},
                )
                raise ImportError(
                    "HybridChunker not available. Install with: pip install 'docling-core[chunking]'"
                ) from e
        
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
        
        try:
            chunks: List[DoclingChunk] = []
            chunk_index = 0
            
            # Use HybridChunker to split document
            # It returns an iterator of chunk items
            for chunk_item in chunker.chunk(docling_doc):
                # Extract text and metadata from chunk
                chunk_text = chunk_item.text if hasattr(chunk_item, 'text') else str(chunk_item)
                
                if not chunk_text or not chunk_text.strip():
                    continue
                
                # Build metadata for this chunk
                chunk_meta = {
                    **doc_metadata,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "docling_hybrid",
                }
                
                # Add structural metadata if available
                if self.include_metadata and hasattr(chunk_item, 'meta'):
                    chunk_meta.update(chunk_item.meta)
                
                # Try to extract heading context
                if hasattr(chunk_item, 'headings'):
                    chunk_meta["headings"] = chunk_item.headings
                
                # Try to extract page numbers
                if hasattr(chunk_item, 'page_no'):
                    chunk_meta["page"] = chunk_item.page_no
                elif hasattr(chunk_item, 'pages'):
                    chunk_meta["pages"] = chunk_item.pages
                
                chunks.append(DoclingChunk(text=chunk_text, meta=chunk_meta))
                chunk_index += 1
            
            LOGGER.info(
                "docling_chunking_complete",
                extra={
                    "doc_id": doc_metadata.get("doc_id", "unknown"),
                    "chunks": len(chunks),
                },
            )
            
            return chunks
            
        except Exception as exc:
            LOGGER.exception(
                "docling_chunking_failed",
                extra={"doc_id": doc_metadata.get("doc_id", "unknown")},
            )
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
            chunk_index = 0
            
            for chunk_item in chunker.chunk(docling_doc):
                chunk_text = chunk_item.text if hasattr(chunk_item, 'text') else str(chunk_item)
                
                if not chunk_text or not chunk_text.strip():
                    continue
                
                chunk_meta = {
                    **doc_metadata,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "docling_hybrid",
                }
                
                if self.include_metadata and hasattr(chunk_item, 'meta'):
                    chunk_meta.update(chunk_item.meta)
                
                if hasattr(chunk_item, 'headings'):
                    chunk_meta["headings"] = chunk_item.headings
                
                if hasattr(chunk_item, 'page_no'):
                    chunk_meta["page"] = chunk_item.page_no
                elif hasattr(chunk_item, 'pages'):
                    chunk_meta["pages"] = chunk_item.pages
                
                yield DoclingChunk(text=chunk_text, meta=chunk_meta)
                chunk_index += 1
                
        except Exception as exc:
            LOGGER.exception(
                "docling_chunking_failed",
                extra={"doc_id": doc_metadata.get("doc_id", "unknown")},
            )
            raise


def chunk_docling_document(
    docling_doc,
    doc_metadata: Optional[Dict[str, Any]] = None,
    max_tokens: int = 512,
    overlap: int = 75,
) -> List[DoclingChunk]:
    """Convenience function to chunk a Docling document.
    
    Args:
        docling_doc: DoclingDocument object from Docling parser
        doc_metadata: Additional metadata to include in chunks
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks
        
    Returns:
        List of DoclingChunk objects
    """
    chunker = DoclingChunker(
        max_tokens=max_tokens,
        overlap=overlap,
    )
    return chunker.chunk_document(docling_doc, doc_metadata)


__all__ = ["DoclingChunker", "DoclingChunk", "chunk_docling_document"]

