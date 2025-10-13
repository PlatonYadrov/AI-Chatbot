"""Cross-encoder reranking service using LangChain components."""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import LangChain reranking service
try:
    from .langchain_reranker import LangChainReranker, rerank_candidates as langchain_rerank_candidates
    LANGCHAIN_RERANKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain reranking not available: {e}")
    LANGCHAIN_RERANKING_AVAILABLE = False


@dataclass
class RerankResult:
    """Result of reranking operation."""
    query: str
    candidates: List[Dict[str, Any]]
    scores: List[float]
    reranked_candidates: List[Dict[str, Any]]


class CrossEncoderReranker:
    """Cross-encoder reranking service using LangChain components."""
    
    def __init__(
        self,
        model_name: str = "langchain-compression",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        cache_dir: Optional[str] = None
    ):
        """Initialize cross-encoder reranker using LangChain.
        
        Args:
            model_name: Model name (for compatibility, uses LangChain)
            device: Device (for compatibility, ignored)
            max_length: Max length (for compatibility, ignored)
            batch_size: Batch size (for compatibility, ignored)
            cache_dir: Cache dir (for compatibility, ignored)
        """
        self.model_name = model_name
        self.device = device or "auto"
        
        logger.info(f"Initializing CrossEncoderReranker with LangChain backend")
        
        # Initialize LangChain reranker
        if LANGCHAIN_RERANKING_AVAILABLE:
            try:
                self.langchain_reranker = LangChainReranker()
                logger.info("Successfully initialized LangChain reranker")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain reranker: {e}")
                self.langchain_reranker = None
        else:
            logger.warning("LangChain reranking not available")
            self.langchain_reranker = None
    
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
            candidates: List of candidate documents with 'text' field
            top_k: Number of top results to return (None for all)
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
        
        # Use LangChain reranker if available
        if self.langchain_reranker is not None:
            return self.langchain_reranker.rerank_candidates(query, candidates, top_k, return_scores)
        else:
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
        """Rerank candidates and return with metadata.
        
        Args:
            query: User query string
            candidates: List of candidate documents
            top_k: Number of top results to return
            include_metadata: Whether to include reranking metadata
            
        Returns:
            List of reranked candidates with optional metadata
        """
        # Use LangChain reranker if available
        if self.langchain_reranker is not None:
            return self.langchain_reranker.rerank_with_metadata(query, candidates, top_k, include_metadata)
        
        # Fallback implementation
        result = self.rerank_candidates(query, candidates, top_k, return_scores=True)
        
        reranked_with_metadata = []
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            enhanced_candidate = candidate.copy()
            
            if include_metadata:
                enhanced_candidate.update({
                    'rerank_score': float(score),
                    'rerank_rank': i + 1,
                    'rerank_model': self.model_name,
                    'rerank_method': 'langchain' if self.langchain_reranker else 'text_similarity'
                })
            
            reranked_with_metadata.append(enhanced_candidate)
        
        return reranked_with_metadata


# Global instance for easy import
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        model_name = os.getenv("CROSS_ENCODER_MODEL", "langchain-compression")
        device = os.getenv("CROSS_ENCODER_DEVICE", None)
        
        _reranker_instance = CrossEncoderReranker(
            model_name=model_name,
            device=device
        )
    
    return _reranker_instance


def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convenience function for reranking candidates using LangChain.
    
    Args:
        query: User query string
        candidates: List of candidate documents
        top_k: Number of top results to return
        
    Returns:
        List of reranked candidates with metadata
    """
    # Try LangChain reranking first
    if LANGCHAIN_RERANKING_AVAILABLE:
        try:
            return langchain_rerank_candidates(query, candidates, top_k)
        except Exception as e:
            logger.warning(f"LangChain reranking failed, using fallback: {e}")
    
    # Fallback to basic reranker
    reranker = get_reranker()
    return reranker.rerank_with_metadata(query, candidates, top_k)


# Backward compatibility
def rerank_candidates_legacy(query: str, candidates):
    """Legacy function for backward compatibility."""
    if not candidates:
        return []
    
    # Convert to expected format
    formatted_candidates = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            formatted_candidates.append(candidate)
        else:
            # Assume it's a Document object with page_content
            formatted_candidates.append({
                'text': getattr(candidate, 'page_content', str(candidate)),
                'metadata': getattr(candidate, 'metadata', {})
            })
    
    return rerank_candidates(query, formatted_candidates)
