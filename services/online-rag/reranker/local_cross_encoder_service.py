#!/usr/bin/env python3
"""Local Cross-Encoder reranking service for offline RAG pipeline."""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

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
    """Result of reranking operation with metrics."""
    query: str
    candidates: List[Dict[str, Any]]
    scores: List[float]
    reranked_candidates: List[Dict[str, Any]]
    processing_time_ms: float = 0.0
    model_name: str = ""
    method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "num_candidates": len(self.candidates),
            "num_reranked": len(self.reranked_candidates),
            "top_score": max(self.scores) if self.scores else 0.0,
            "bottom_score": min(self.scores) if self.scores else 0.0,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "method": self.method
        }


class LocalCrossEncoderReranker:
    """
    Local cross-encoder reranker with advanced features.
    
    Features:
        - Multilingual support with mMiniLM models
        - Batch processing for efficiency
        - Score normalization
        - Detailed metrics and logging
        - Fallback to text similarity when model unavailable
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-mMiniLM-L-12-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        offline_mode: bool = True,
        normalize_scores: bool = True
    ):
        """
        Initialize local cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: cross-encoder/ms-marco-mMiniLM-L-12-v2 (multilingual)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for the model
            batch_size: Batch size for processing (32 optimal for L4 GPU)
            cache_dir: Directory to cache model files
            offline_mode: Whether to run in offline mode (no internet required)
            normalize_scores: Whether to normalize scores to [0, 1] range
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.offline_mode = offline_mode
        self.normalize_scores = normalize_scores
        self.model = None
        
        if device is None:
            self.device = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        
        self._load_model(cache_dir)
    
    def _load_model(self, cache_dir: Optional[str] = None) -> None:
        """Load cross-encoder model with error handling."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("❌ Transformers not available, using fallback reranking")
            return
        
        try:
            logger.info(f"🔄 Loading cross-encoder model: {self.model_name}")
            logger.info(f"   Device: {self.device}, Offline: {self.offline_mode}")
            
            start_time = time.time()
            
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                cache_folder=cache_dir,
                trust_remote_code=False
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Successfully loaded cross-encoder in {load_time:.2f}s")
            logger.info(f"   Model parameters: ~{self._estimate_parameters()}M")
            
        except Exception as e:
            logger.error(f"❌ Failed to load cross-encoder model {self.model_name}: {e}")
            logger.warning("   Falling back to text-based reranking")
            self.model = None
    
    def _estimate_parameters(self) -> int:
        """Estimate model parameter count in millions."""
        if self.model is None:
            return 0
        try:
            if hasattr(self.model, 'model'):
                params = sum(p.numel() for p in self.model.model.parameters())
                return params // 1_000_000
        except:
            pass
        return 12  # Default estimate for MiniLM-L-12
    
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        score_threshold: Optional[float] = None
    ) -> RerankResult:
        """
        Rerank candidates using cross-encoder or fallback method.
        
        Args:
            query: User query string
            candidates: List of candidate documents with 'text' field
            top_k: Number of top results to return (None for all)
            return_scores: Whether to include scores in results
            score_threshold: Minimum score threshold for filtering (None for no filter)
            
        Returns:
            RerankResult with reranked candidates, scores, and metrics
        """
        start_time = time.time()
        
        if not candidates:
            return RerankResult(
                query=query,
                candidates=[],
                scores=[],
                reranked_candidates=[],
                processing_time_ms=0.0,
                model_name=self.model_name,
                method="none"
            )
        
        logger.info(f"🔍 Reranking {len(candidates)} candidates for query: '{query[:80]}...'")
        
        if self.model is not None:
            result = self._rerank_with_cross_encoder(query, candidates, top_k, return_scores, score_threshold)
        else:
            result = self._rerank_with_fallback(query, candidates, top_k, return_scores)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Reranking completed in {result.processing_time_ms:.2f}ms")
        logger.info(f"   Results: {len(result.reranked_candidates)}/{len(candidates)} candidates")
        if result.scores:
            logger.info(f"   Scores: top={result.scores[0]:.4f}, bottom={result.scores[-1]:.4f}")
        
        return result
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int],
        return_scores: bool,
        score_threshold: Optional[float] = None
    ) -> RerankResult:
        """
        Rerank using actual cross-encoder model with advanced features.
        
        Features:
            - Batch processing for efficiency
            - Score normalization
            - Threshold filtering
            - Detailed logging
        """
        pairs = []
        valid_candidates = []
        
        for i, candidate in enumerate(candidates):
            text = candidate.get('text', candidate.get('page_content', ''))
            if not text or not text.strip():
                logger.debug(f"⚠️  Skipping candidate {i}: missing text field")
                continue
            
            text_truncated = text[:self.max_length * 4]
            pairs.append([query, text_truncated])
            valid_candidates.append(candidate)
        
        if not pairs:
            logger.warning("❌ No valid candidates found for reranking")
            return RerankResult(
                query=query,
                candidates=candidates,
                scores=[],
                reranked_candidates=[],
                model_name=self.model_name,
                method="cross_encoder_failed"
            )
        
        logger.debug(f"📊 Processing {len(pairs)} query-document pairs")
        logger.debug(f"   Batch size: {self.batch_size}, Device: {self.device}")
        
        try:
            inference_start = time.time()
            
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            inference_time = (time.time() - inference_start) * 1000
            logger.debug(f"⚡ Inference completed in {inference_time:.2f}ms")
            logger.debug(f"   Throughput: {len(pairs) / (inference_time / 1000):.1f} pairs/sec")
            
            scores = raw_scores.tolist() if hasattr(raw_scores, 'tolist') else list(raw_scores)
            
            if self.normalize_scores:
                scores = self._normalize_scores(scores)
                logger.debug(f"📈 Scores normalized to [0, 1] range")
            
        except Exception as e:
            logger.error(f"❌ Cross-encoder prediction failed: {e}")
            return self._rerank_with_fallback(query, candidates, top_k, return_scores)
        
        scored_candidates = list(zip(scores, valid_candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if score_threshold is not None:
            before_filter = len(scored_candidates)
            scored_candidates = [(s, c) for s, c in scored_candidates if s >= score_threshold]
            logger.debug(f"🔽 Score threshold {score_threshold}: {before_filter} → {len(scored_candidates)} candidates")
        
        if top_k is not None and len(scored_candidates) > top_k:
            scored_candidates = scored_candidates[:top_k]
        
        reranked_scores = [score for score, _ in scored_candidates]
        reranked_candidates = [candidate for _, candidate in scored_candidates]
        
        return RerankResult(
            query=query,
            candidates=candidates,
            scores=reranked_scores if return_scores else [],
            reranked_candidates=reranked_candidates,
            model_name=self.model_name,
            method="cross_encoder"
        )
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max scaling.
        
        Args:
            scores: Raw scores from cross-encoder
            
        Returns:
            Normalized scores in [0, 1] range
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _rerank_with_fallback(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int],
        return_scores: bool
    ) -> RerankResult:
        """
        Fallback reranking using improved text similarity.
        
        Uses multiple signals:
            - Jaccard similarity (word overlap)
            - Exact phrase matching
            - Metadata matching
            - Length penalty for very short/long documents
        """
        logger.warning("⚠️  Using fallback text-based reranking")
        
        query_words = set(query.lower().split())
        query_length = len(query)
        scored_candidates = []
        
        for candidate in candidates:
            text = candidate.get('text', candidate.get('page_content', ''))
            if not text or not text.strip():
                continue
            
            text_lower = text.lower()
            text_words = set(text_lower.split())
            
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            jaccard_score = overlap / total_words if total_words > 0 else 0.0
            
            score = jaccard_score
            
            if query.lower() in text_lower:
                score += 0.5
                query_position = text_lower.find(query.lower())
                position_bonus = max(0, 0.2 * (1 - query_position / len(text)))
                score += position_bonus
            
            metadata = candidate.get('metadata', {})
            if any(query.lower() in str(value).lower() for value in metadata.values()):
                score += 0.3
            
            text_length = len(text)
            if text_length < 50:
                score *= 0.7
            elif text_length > 5000:
                score *= 0.9
            
            score = min(score, 1.0)
            
            scored_candidates.append((score, candidate))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]
        
        reranked_scores = [score for score, _ in scored_candidates]
        reranked_candidates = [candidate for _, candidate in scored_candidates]
        
        if reranked_scores:
            logger.info(f"📊 Fallback reranking: top={reranked_scores[0]:.4f}, "
                       f"bottom={reranked_scores[-1]:.4f}")
        
        return RerankResult(
            query=query,
            candidates=candidates,
            scores=reranked_scores if return_scores else [],
            reranked_candidates=reranked_candidates,
            model_name="text_similarity_fallback",
            method="fallback"
        )
    
    def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        include_metadata: bool = True,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return with detailed metadata.
        
        Args:
            query: User query string
            candidates: List of candidate documents
            top_k: Number of top results to return
            include_metadata: Whether to include reranking metadata
            score_threshold: Minimum score threshold for filtering
            
        Returns:
            List of reranked candidates with optional reranking metadata
        """
        result = self.rerank_candidates(
            query, 
            candidates, 
            top_k, 
            return_scores=True,
            score_threshold=score_threshold
        )
        
        reranked_with_metadata = []
        for i, (candidate, score) in enumerate(zip(result.reranked_candidates, result.scores)):
            enhanced_candidate = candidate.copy()
            
            if include_metadata:
                enhanced_candidate.update({
                    'rerank_score': float(score),
                    'rerank_rank': i + 1,
                    'rerank_model': result.model_name,
                    'rerank_method': result.method,
                    'rerank_processing_time_ms': result.processing_time_ms
                })
            
            reranked_with_metadata.append(enhanced_candidate)
        
        return reranked_with_metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "offline_mode": self.offline_mode,
            "normalize_scores": self.normalize_scores,
            "model_loaded": self.model is not None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "cuda_available": TRANSFORMERS_AVAILABLE and torch.cuda.is_available()
        }


# Global instance for easy import
_reranker_instance: Optional[LocalCrossEncoderReranker] = None


def get_local_reranker() -> LocalCrossEncoderReranker:
    """
    Get or create global local reranker instance.
    
    Environment variables:
        CROSS_ENCODER_MODEL: Model name (default: cross-encoder/ms-marco-mMiniLM-L-12-v2)
        CROSS_ENCODER_DEVICE: Device (default: auto)
        CROSS_ENCODER_CACHE_DIR: Cache directory (default: None)
        CROSS_ENCODER_OFFLINE_MODE: Offline mode (default: true)
        CROSS_ENCODER_BATCH_SIZE: Batch size (default: 32)
        CROSS_ENCODER_MAX_LENGTH: Max sequence length (default: 512)
        CROSS_ENCODER_NORMALIZE: Normalize scores (default: true)
    
    Returns:
        LocalCrossEncoderReranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-mMiniLM-L-12-v2")
        device = os.getenv("CROSS_ENCODER_DEVICE", None)
        cache_dir = os.getenv("CROSS_ENCODER_CACHE_DIR", None)
        offline_mode = os.getenv("CROSS_ENCODER_OFFLINE_MODE", "true").lower() == "true"
        batch_size = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32"))
        max_length = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
        normalize_scores = os.getenv("CROSS_ENCODER_NORMALIZE", "true").lower() == "true"
        
        logger.info("🚀 Initializing global Cross-Encoder reranker")
        
        _reranker_instance = LocalCrossEncoderReranker(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            offline_mode=offline_mode,
            batch_size=batch_size,
            max_length=max_length,
            normalize_scores=normalize_scores
        )
        
        info = _reranker_instance.get_model_info()
        logger.info(f"📋 Reranker info: {info}")
    
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
    """
    Get detailed information about reranking capabilities.
    
    Returns:
        Dictionary with reranking system information
    """
    reranker = get_local_reranker()
    info = reranker.get_model_info()
    
    info.update({
        "available_models": [
            "cross-encoder/ms-marco-mMiniLM-L-12-v2",  # Multilingual, 12 layers
            "cross-encoder/ms-marco-MiniLM-L-6-v2",    # Faster, 6 layers
            "cross-encoder/ms-marco-electra-base",     # Higher quality
        ],
        "recommended_model": "cross-encoder/ms-marco-mMiniLM-L-12-v2",
        "status": "ready" if reranker.model is not None else "fallback"
    })
    
    return info
