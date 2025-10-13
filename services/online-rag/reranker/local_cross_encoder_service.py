#!/usr/bin/env python3
"""Local Cross-Encoder reranking service for offline RAG pipeline."""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    import numpy as np
    from sentence_transformers import CrossEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class RerankResult:
    """Result of reranking operation."""
    query: str
    candidates: List[Dict[str, Any]]
    scores: List[float]
    reranked_candidates: List[Dict[str, Any]]


class LocalCrossEncoderReranker:
    """Local cross-encoder reranker that works offline."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        cache_dir: Optional[str] = None,
        offline_mode: bool = True
    ):
        """Initialize local cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for the model
            batch_size: Batch size for processing
            cache_dir: Directory to cache model files
            offline_mode: Whether to run in offline mode (no internet required)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.offline_mode = offline_mode
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing LocalCrossEncoderReranker with model={model_name}, device={self.device}, offline={offline_mode}")
        
        # Initialize model if transformers available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.model = CrossEncoder(
                    model_name,
                    max_length=max_length,
                    device=self.device,
                    cache_folder=cache_dir
                )
                logger.info(f"Successfully loaded cross-encoder model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
                self.model = None
        else:
            logger.warning("Transformers not available, using fallback reranking")
            self.model = None
    
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> RerankResult:
        """Rerank candidates using cross-encoder or fallback method.
        
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
        
        # Use cross-encoder if available
        if self.model is not None:
            return self._rerank_with_cross_encoder(query, candidates, top_k, return_scores)
        else:
            # Fallback to simple text-based reranking
            return self._rerank_with_fallback(query, candidates, top_k, return_scores)
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int],
        return_scores: bool
    ) -> RerankResult:
        """Rerank using actual cross-encoder model."""
        # Prepare query-document pairs
        pairs = []
        valid_candidates = []
        
        for candidate in candidates:
            text = candidate.get('text', candidate.get('page_content', ''))
            if not text:
                logger.warning(f"Candidate missing text field: {candidate}")
                continue
            pairs.append([query, text])
            valid_candidates.append(candidate)
        
        if not pairs:
            logger.warning("No valid candidates found for reranking")
            return RerankResult(
                query=query,
                candidates=candidates,
                scores=[],
                reranked_candidates=[]
            )
        
        # Get relevance scores
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            # Fallback to simple reranking
            return self._rerank_with_fallback(query, candidates, top_k, return_scores)
        
        # Create score-candidate pairs and sort by score (descending)
        scored_candidates = list(zip(scores, valid_candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Apply top_k filter
        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]
        
        # Extract results
        reranked_scores = [score for score, _ in scored_candidates]
        reranked_candidates = [candidate for _, candidate in scored_candidates]
        
        logger.info(f"Cross-encoder reranking completed. Top score: {reranked_scores[0]:.4f}, "
                   f"Bottom score: {reranked_scores[-1]:.4f}")
        
        return RerankResult(
            query=query,
            candidates=candidates,
            scores=reranked_scores if return_scores else [],
            reranked_candidates=reranked_candidates
        )
    
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
        result = self.rerank_candidates(query, candidates, top_k, return_scores=True)
        
        reranked_with_metadata = []
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            enhanced_candidate = candidate.copy()
            
            if include_metadata:
                enhanced_candidate.update({
                    'rerank_score': float(score),
                    'rerank_rank': i + 1,
                    'rerank_model': self.model_name if self.model else 'fallback',
                    'rerank_method': 'cross_encoder' if self.model else 'text_similarity'
                })
            
            reranked_with_metadata.append(enhanced_candidate)
        
        return reranked_with_metadata


# Global instance for easy import
_reranker_instance: Optional[LocalCrossEncoderReranker] = None


def get_local_reranker() -> LocalCrossEncoderReranker:
    """Get global local reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        device = os.getenv("CROSS_ENCODER_DEVICE", None)
        cache_dir = os.getenv("CROSS_ENCODER_CACHE_DIR", None)
        offline_mode = os.getenv("CROSS_ENCODER_OFFLINE_MODE", "true").lower() == "true"
        
        _reranker_instance = LocalCrossEncoderReranker(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            offline_mode=offline_mode
        )
    
    return _reranker_instance


def rerank_candidates_local(query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convenience function for local reranking candidates.
    
    Args:
        query: User query string
        candidates: List of candidate documents
        top_k: Number of top results to return
        
    Returns:
        List of reranked candidates with metadata
    """
    reranker = get_local_reranker()
    return reranker.rerank_with_metadata(query, candidates, top_k)


# Backward compatibility
def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Backward compatibility function."""
    return rerank_candidates_local(query, candidates, top_k)


def is_reranking_available() -> bool:
    """Check if reranking is available."""
    return TRANSFORMERS_AVAILABLE or True  # Always available with fallback


def get_reranking_info() -> Dict[str, Any]:
    """Get information about reranking capabilities."""
    return {
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "device": get_local_reranker().device,
        "model": get_local_reranker().model_name,
        "offline_mode": get_local_reranker().offline_mode,
        "method": "cross_encoder" if get_local_reranker().model else "text_similarity"
    }
