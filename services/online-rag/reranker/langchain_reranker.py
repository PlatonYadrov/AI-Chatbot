"""LangChain-based Cross-Encoder reranking service."""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import LangChain reranking components
try:
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.retrievers.document_compressors import DocumentCompressorPipeline
    from langchain.retrievers.document_compressors import LLMChainFilter
    from langchain.llms.base import LLM
    from langchain.embeddings.base import Embeddings
    LANGCHAIN_RERANKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain reranking components not available: {e}")
    LANGCHAIN_RERANKING_AVAILABLE = False


@dataclass
class RerankResult:
    """Result of reranking operation."""
    query: str
    candidates: List[Dict[str, Any]]
    scores: List[float]
    reranked_candidates: List[Dict[str, Any]]


class LangChainReranker:
    """LangChain-based reranking service using built-in components."""
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        embeddings: Optional[Embeddings] = None,
        similarity_threshold: float = 0.76,
        k: int = 4,
        use_compression: bool = True
    ):
        """Initialize LangChain reranker.
        
        Args:
            llm: LangChain LLM instance for filtering
            embeddings: LangChain embeddings instance
            similarity_threshold: Threshold for similarity filtering
            k: Number of documents to return
            use_compression: Whether to use compression pipeline
        """
        self.llm = llm
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.k = k
        self.use_compression = use_compression
        
        logger.info(f"Initializing LangChainReranker with compression={use_compression}")
        
        # Initialize compression pipeline if components available
        if LANGCHAIN_RERANKING_AVAILABLE and use_compression:
            self._setup_compression_pipeline()
        else:
            logger.warning("LangChain reranking components not available, using fallback")
            self.compression_pipeline = None
    
    def _setup_compression_pipeline(self):
        """Setup LangChain compression pipeline."""
        try:
            compressors = []
            
            # Add embeddings filter if embeddings available
            if self.embeddings:
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.embeddings,
                    similarity_threshold=self.similarity_threshold,
                    k=self.k
                )
                compressors.append(embeddings_filter)
            
            # Add LLM filter if LLM available
            if self.llm:
                llm_filter = LLMChainFilter.from_llm(self.llm)
                compressors.append(llm_filter)
            
            if compressors:
                self.compression_pipeline = DocumentCompressorPipeline(
                    transformers=compressors
                )
                logger.info(f"Compression pipeline setup with {len(compressors)} compressors")
            else:
                logger.warning("No compressors available, using fallback")
                self.compression_pipeline = None
                
        except Exception as e:
            logger.error(f"Failed to setup compression pipeline: {e}")
            self.compression_pipeline = None
    
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> RerankResult:
        """Rerank candidates using LangChain components.
        
        Args:
            query: User query string
            candidates: List of candidate documents
            top_k: Number of top results to return
            return_scores: Whether to include scores in results
            
        Returns:
            RerankResult with reranked candidates and scores
        """
        if not candidates:
            return RerankResult(
                query=query,
                candidates=[],
                scores=[],
                reranked_candidates=[]
            )
        
        logger.info(f"Reranking {len(candidates)} candidates for query: {query[:100]}...")
        
        # Use LangChain compression if available
        if self.compression_pipeline and LANGCHAIN_RERANKING_AVAILABLE:
            return self._rerank_with_langchain(query, candidates, top_k, return_scores)
        else:
            return self._rerank_with_fallback(query, candidates, top_k, return_scores)
    
    def _rerank_with_langchain(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int],
        return_scores: bool
    ) -> RerankResult:
        """Rerank using LangChain compression pipeline."""
        try:
            # Convert candidates to LangChain Document format
            from langchain.schema import Document
            
            docs = []
            for candidate in candidates:
                text = candidate.get('text', candidate.get('page_content', ''))
                metadata = candidate.get('metadata', {})
                docs.append(Document(page_content=text, metadata=metadata))
            
            # Apply compression pipeline
            compressed_docs = self.compression_pipeline.compress_documents(
                docs, query
            )
            
            # Convert back to our format
            reranked_candidates = []
            scores = []
            
            for doc in compressed_docs:
                candidate = {
                    'text': doc.page_content,
                    'metadata': doc.metadata
                }
                reranked_candidates.append(candidate)
                
                # Generate score based on position (LangChain doesn't provide explicit scores)
                score = 1.0 - (len(scores) * 0.1)  # Decreasing scores
                scores.append(max(score, 0.1))
            
            # Apply top_k filter
            if top_k is not None:
                reranked_candidates = reranked_candidates[:top_k]
                scores = scores[:top_k]
            
            logger.info(f"LangChain reranking completed. Returned {len(reranked_candidates)} documents")
            
            return RerankResult(
                query=query,
                candidates=candidates,
                scores=scores if return_scores else [],
                reranked_candidates=reranked_candidates
            )
            
        except Exception as e:
            logger.error(f"LangChain reranking failed: {e}")
            return self._rerank_with_fallback(query, candidates, top_k, return_scores)
    
    def _rerank_with_fallback(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int],
        return_scores: bool
    ) -> RerankResult:
        """Fallback reranking using simple text similarity."""
        logger.info("Using fallback text-based reranking")
        
        query_words = set(query.lower().split())
        scored_candidates = []
        
        for candidate in candidates:
            text = candidate.get('text', candidate.get('page_content', ''))
            if not text:
                continue
                
            # Simple word overlap scoring
            text_words = set(text.lower().split())
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            
            # Jaccard similarity
            score = overlap / total_words if total_words > 0 else 0.0
            
            # Boost score for exact phrase matches
            if query.lower() in text.lower():
                score += 0.5
            
            # Boost score for title/header matches
            metadata = candidate.get('metadata', {})
            if any(query.lower() in str(value).lower() for value in metadata.values()):
                score += 0.3
            
            scored_candidates.append((score, candidate))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Apply top_k filter
        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]
        
        # Extract results
        reranked_scores = [score for score, _ in scored_candidates]
        reranked_candidates = [candidate for _, candidate in scored_candidates]
        
        logger.info(f"Fallback reranking completed. Top score: {reranked_scores[0]:.4f}, "
                   f"Bottom score: {reranked_scores[-1]:.4f}")
        
        return RerankResult(
            query=query,
            candidates=candidates,
            scores=reranked_scores if return_scores else [],
            reranked_candidates=reranked_candidates
        )
    
    def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Rerank candidates and return with metadata."""
        result = self.rerank_candidates(query, candidates, top_k, return_scores=True)
        
        reranked_with_metadata = []
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            enhanced_candidate = candidate.copy()
            
            if include_metadata:
                enhanced_candidate.update({
                    'rerank_score': float(score),
                    'rerank_rank': i + 1,
                    'rerank_model': 'langchain_compression' if self.compression_pipeline else 'fallback',
                    'rerank_method': 'langchain' if self.compression_pipeline else 'text_similarity'
                })
            
            reranked_with_metadata.append(enhanced_candidate)
        
        return reranked_with_metadata


# Global instance for easy import
_reranker_instance: Optional[LangChainReranker] = None


def get_langchain_reranker() -> LangChainReranker:
    """Get global LangChain reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        # Initialize with default components
        llm = None
        embeddings = None
        
        # Try to get LLM from environment
        try:
            from generation.llm_service import get_llm_client
            llm = get_llm_client()
        except Exception as e:
            logger.warning(f"Could not initialize LLM for reranking: {e}")
        
        # Try to get embeddings from environment
        try:
            from api.gateway import TEIEmbeddings
            embeddings = TEIEmbeddings()
        except Exception as e:
            logger.warning(f"Could not initialize embeddings for reranking: {e}")
        
        _reranker_instance = LangChainReranker(
            llm=llm,
            embeddings=embeddings,
            use_compression=True
        )
    
    return _reranker_instance


def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convenience function for reranking candidates using LangChain."""
    reranker = get_langchain_reranker()
    return reranker.rerank_with_metadata(query, candidates, top_k)


def is_reranking_available() -> bool:
    """Check if reranking is available."""
    return LANGCHAIN_RERANKING_AVAILABLE or True  # Always available with fallback


def get_reranking_info() -> Dict[str, Any]:
    """Get information about reranking capabilities."""
    reranker = get_langchain_reranker()
    return {
        "langchain_available": LANGCHAIN_RERANKING_AVAILABLE,
        "compression_pipeline": reranker.compression_pipeline is not None,
        "llm_available": reranker.llm is not None,
        "embeddings_available": reranker.embeddings is not None,
        "method": "langchain" if reranker.compression_pipeline else "text_similarity"
    }
