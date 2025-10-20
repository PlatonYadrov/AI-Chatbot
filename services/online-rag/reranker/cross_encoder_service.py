"""
Cross-encoder reranking service.

This module provides a unified interface for cross-encoder reranking,
prioritizing Infinity service (no torch) with fallback to local implementation.
"""

import os
from typing import List, Dict, Any, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Primary: Infinity reranker (no torch dependency, container-based)
try:
    from .infinity_reranker import (
        InfinityReranker,
        get_infinity_reranker,
        rerank_candidates_infinity,
        RerankResult,
        is_reranking_available as infinity_available,
        get_reranking_info as infinity_info
    )
    INFINITY_RERANKER_AVAILABLE = True
    logger.info("✅ Infinity reranker available (no torch in Python)")
except ImportError as e:
    logger.warning(f"⚠️  Infinity reranker not available: {e}")
    INFINITY_RERANKER_AVAILABLE = False

# Fallback: Local torch-based reranker (if torch is installed)
try:
    from .local_cross_encoder_service import (
        LocalCrossEncoderReranker,
        get_local_reranker,
        rerank_candidates_local
    )
    LOCAL_RERANKER_AVAILABLE = True
    logger.info("✅ Local torch reranker available (fallback)")
except ImportError as e:
    logger.warning(f"⚠️  Local reranker not available: {e}")
    LOCAL_RERANKER_AVAILABLE = False


class CrossEncoderReranker:
    """
    Unified cross-encoder reranking service.
    
    Priority order:
        1. Infinity service (container-based, no torch in Python)
        2. Local torch reranker (fallback if Infinity unavailable)
        3. Simple text similarity (fallback if both unavailable)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        prefer_infinity: bool = True
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Model name (used for Infinity or local)
            device: Device for local reranker ('cuda', 'cpu', or None)
            max_length: Max sequence length for local reranker
            batch_size: Batch size for local reranker
            cache_dir: Cache directory for local reranker
            prefer_infinity: Whether to prefer Infinity over local (default: True)
        """
        self.model_name = model_name
        self.device = device or "auto"
        self.prefer_infinity = prefer_infinity
        
        logger.info(f"🚀 Initializing CrossEncoderReranker")
        logger.info(f"   Prefer Infinity: {prefer_infinity}")
        logger.info(f"   Model: {model_name}")
        
        # Try Infinity first if preferred
        self.infinity_reranker = None
        self.local_reranker = None
        
        if prefer_infinity and INFINITY_RERANKER_AVAILABLE:
            try:
                self.infinity_reranker = get_infinity_reranker()
                logger.info("✅ Using Infinity reranker (no torch in Python)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Infinity reranker: {e}")
        
        # Fallback to local if Infinity not available
        if self.infinity_reranker is None and LOCAL_RERANKER_AVAILABLE:
            try:
                self.local_reranker = LocalCrossEncoderReranker(
                    model_name=model_name,
                    device=device,
                    max_length=max_length,
                    batch_size=batch_size,
                    cache_dir=cache_dir
                )
                logger.info("✅ Using local torch reranker (fallback)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize local reranker: {e}")
        
        if self.infinity_reranker is None and self.local_reranker is None:
            logger.warning("⚠️  No rerankers available, will use text similarity fallback")
    
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        score_threshold: Optional[float] = None
    ) -> "RerankResult":
        """
        Rerank candidates using cross-encoder model.
        
        Args:
            query: User query string
            candidates: List of candidate documents with 'text' field
            top_k: Number of top results to return (None for all)
            return_scores: Whether to include scores in results
            score_threshold: Minimum score threshold for filtering
            
        Returns:
            RerankResult with reranked candidates and scores
        """
        if not candidates:
            if INFINITY_RERANKER_AVAILABLE or LOCAL_RERANKER_AVAILABLE:
                return RerankResult(
                    query=query,
                    candidates=[],
                    scores=[],
                    reranked_candidates=[],
                    processing_time_ms=0.0,
                    model_name=self.model_name,
                    method="none"
                )
            else:
                return self._create_empty_result(query)
        
        # Priority 1: Infinity reranker
        if self.infinity_reranker is not None:
            return self.infinity_reranker.rerank_candidates(
                query,
                candidates,
                top_k,
                return_scores,
                score_threshold
            )
        
        # Priority 2: Local torch reranker
        elif self.local_reranker is not None:
            return self.local_reranker.rerank_candidates(
                query, 
                candidates, 
                top_k, 
                return_scores,
                score_threshold
            )
        
        # Priority 3: Text similarity fallback
        else:
            return self._rerank_with_fallback(query, candidates, top_k, return_scores)
    
    def _create_empty_result(self, query: str) -> Dict[str, Any]:
        """Create empty result when RerankResult is not available."""
        return {
            "query": query,
            "candidates": [],
            "scores": [],
            "reranked_candidates": []
        }
    
    def _rerank_with_fallback(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int],
        return_scores: bool
    ) -> Dict[str, Any]:
        """Fallback reranking using simple text similarity."""
        logger.warning("⚠️  Using fallback text-based reranking")
        
        query_words = set(query.lower().split())
        scored_candidates = []
        
        for candidate in candidates:
            text = candidate.get('text', candidate.get('page_content', ''))
            if not text:
                continue
            
            text_words = set(text.lower().split())
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            score = overlap / total_words if total_words > 0 else 0.0
            
            if query.lower() in text.lower():
                score += 0.5
            
            metadata = candidate.get('metadata', {})
            if any(query.lower() in str(value).lower() for value in metadata.values()):
                score += 0.3
            
            scored_candidates.append((score, candidate))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]
        
        reranked_scores = [score for score, _ in scored_candidates]
        reranked_candidates = [candidate for _, candidate in scored_candidates]
        
        if reranked_scores:
            logger.info(f"📊 Fallback: top={reranked_scores[0]:.4f}, "
                       f"bottom={reranked_scores[-1]:.4f}")
        
        return {
            "query": query,
            "candidates": candidates,
            "scores": reranked_scores if return_scores else [],
            "reranked_candidates": reranked_candidates
        }
    
    def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        include_metadata: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return with metadata.
        
        Args:
            query: User query string
            candidates: List of candidate documents
            top_k: Number of top results to return
            include_metadata: Whether to include reranking metadata
            score_threshold: Minimum score threshold for filtering
            
        Returns:
            List of reranked candidates with optional metadata
        """
        # Use Infinity if available
        if self.infinity_reranker is not None:
            return self.infinity_reranker.rerank_with_metadata(
                query,
                candidates,
                top_k,
                include_metadata,
                score_threshold
            )
        
        # Use local reranker if available
        elif self.local_reranker is not None:
            return self.local_reranker.rerank_with_metadata(
                query, 
                candidates, 
                top_k, 
                include_metadata,
                score_threshold
            )
        
        # Fallback implementation
        result = self.rerank_candidates(query, candidates, top_k, return_scores=True)
        
        reranked_with_metadata = []
        reranked_candidates = result.get("reranked_candidates", []) if isinstance(result, dict) else result.reranked_candidates
        scores = result.get("scores", []) if isinstance(result, dict) else result.scores
        
        for i, (candidate, score) in enumerate(zip(reranked_candidates, scores)):
            enhanced_candidate = candidate.copy()
            
            if include_metadata:
                enhanced_candidate.update({
                    'rerank_score': float(score) if score else 0.0,
                    'rerank_rank': i + 1,
                    'rerank_model': self.model_name,
                    'rerank_method': 'text_similarity_fallback'
                })
            
            reranked_with_metadata.append(enhanced_candidate)
        
        return reranked_with_metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        if self.infinity_reranker is not None:
            return self.infinity_reranker.get_model_info()
        elif self.local_reranker is not None:
            return self.local_reranker.get_model_info()
        return {
            "model_name": self.model_name,
            "device": self.device,
            "status": "fallback_text_similarity",
            "infinity_available": INFINITY_RERANKER_AVAILABLE,
            "local_reranker_available": LOCAL_RERANKER_AVAILABLE
        }


# Global instance for easy import
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """
    Get or create global reranker instance.
    
    Environment variables:
        RERANKER_MODEL: Model name (default: BAAI/bge-reranker-v2-m3)
        RERANKER_PREFER_INFINITY: Prefer Infinity over local (default: true)
        CROSS_ENCODER_DEVICE: Device for local reranker (default: auto)
        CROSS_ENCODER_BATCH_SIZE: Batch size for local reranker (default: 32)
    
    Returns:
        CrossEncoderReranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        prefer_infinity = os.getenv("RERANKER_PREFER_INFINITY", "true").lower() == "true"
        device = os.getenv("CROSS_ENCODER_DEVICE", None)
        batch_size = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32"))
        max_length = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
        cache_dir = os.getenv("CROSS_ENCODER_CACHE_DIR", None)
        
        _reranker_instance = CrossEncoderReranker(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            prefer_infinity=prefer_infinity
        )
    
    return _reranker_instance


def rerank_candidates(
    query: str, 
    candidates: List[Dict[str, Any]], 
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for reranking candidates.
    
    Automatically uses best available method:
        1. Infinity service (no torch in Python)
        2. Local torch reranker (if torch installed)
        3. Text similarity fallback
    
    Args:
        query: User query string
        candidates: List of candidate documents
        top_k: Number of top results to return
        score_threshold: Minimum score threshold for filtering
        
    Returns:
        List of reranked candidates with metadata
    """
    # Try Infinity first
    if INFINITY_RERANKER_AVAILABLE:
        try:
            return rerank_candidates_infinity(query, candidates, top_k)
        except Exception as e:
            logger.warning(f"⚠️  Infinity reranking failed: {e}")
    
    # Try local reranker
    if LOCAL_RERANKER_AVAILABLE:
        try:
            return rerank_candidates_local(query, candidates, top_k)
        except Exception as e:
            logger.warning(f"⚠️  Local reranking failed: {e}")
    
    # Fallback to unified reranker
    reranker = get_reranker()
    return reranker.rerank_with_metadata(
        query, 
        candidates, 
        top_k,
        score_threshold=score_threshold
    )


def is_reranking_available() -> bool:
    """Check if any reranking method is available."""
    return INFINITY_RERANKER_AVAILABLE or LOCAL_RERANKER_AVAILABLE or True  # Always available with fallback


def get_reranking_info() -> Dict[str, Any]:
    """Get detailed information about reranking capabilities."""
    if INFINITY_RERANKER_AVAILABLE:
        try:
            return infinity_info()
        except:
            pass
    
    reranker = get_reranker()
    return reranker.get_model_info()


def rerank_candidates_legacy(query: str, candidates):
    """Legacy function for backward compatibility."""
    if not candidates:
        return []
    
    formatted_candidates = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            formatted_candidates.append(candidate)
        else:
            formatted_candidates.append({
                'text': getattr(candidate, 'page_content', str(candidate)),
                'metadata': getattr(candidate, 'metadata', {})
            })
    
    return rerank_candidates(query, formatted_candidates)
