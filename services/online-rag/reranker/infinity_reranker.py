"""TEI/Infinity-based reranker client (no torch dependency).

Works with both:
- HuggingFace Text Embeddings Inference (TEI) 1.5+ with reranking support
- Infinity service

Both use compatible API formats.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

import requests


@dataclass
class RerankResult:
    """Result of reranking operation."""
    query: str
    candidates: List[Dict[str, Any]]
    scores: List[float]
    reranked_candidates: List[Dict[str, Any]]
    processing_time_ms: float = 0.0
    model_name: str = ""
    method: str = "infinity"
    
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


class InfinityReranker:
    """
    Reranker using TEI or Infinity inference service.
    
    Compatible with:
        - HuggingFace Text Embeddings Inference (TEI) 1.5+ (recommended)
        - Infinity service
    
    Features:
        - Zero torch dependency in Python code
        - HTTP-based inference (all heavy lifting in container)
        - Support for any cross-encoder model
        - Optimized batching and throughput
        - Easy model swapping via docker-compose
        - No SELinux/AppArmor issues
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Infinity reranker client.
        
        Args:
            base_url: Infinity service URL (default from env RERANKER_URL)
            model_name: Model name (for logging and metadata)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = (
            base_url or 
            os.getenv("RERANKER_URL", "http://reranker:80")
        ).rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"🚀 Initialized InfinityReranker")
        logger.info(f"   URL: {self.base_url}")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Timeout: {timeout}s")
        
        self._check_health()
    
    def _check_health(self) -> bool:
        """Check if Infinity service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Reranker service is healthy")
                return True
            else:
                logger.warning(f"⚠️  Reranker health check returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  Reranker service health check failed: {e}")
            logger.warning(f"   Service will be used when available")
        return False
    
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        score_threshold: Optional[float] = None
    ) -> RerankResult:
        """
        Rerank candidates using Infinity service.
        
        Args:
            query: User query string
            candidates: List of candidate documents with 'text' or 'page_content' field
            top_k: Number of top results to return (None for all)
            return_scores: Whether to include scores in results
            score_threshold: Minimum score threshold for filtering
            
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
                method="infinity"
            )
        
        # Extract texts from candidates
        texts = []
        valid_candidates = []
        
        for i, candidate in enumerate(candidates):
            text = candidate.get('text', candidate.get('page_content', ''))
            if not text or not text.strip():
                logger.debug(f"⚠️  Skipping candidate {i}: missing text field")
                continue
            texts.append(text)
            valid_candidates.append(candidate)
        
        if not texts:
            logger.warning("❌ No valid texts for reranking")
            return RerankResult(
                query=query,
                candidates=candidates,
                scores=[],
                reranked_candidates=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_name=self.model_name,
                method="infinity_failed"
            )
        
        logger.info(f"🔍 Reranking {len(texts)} candidates for query: '{query[:80]}...'")
        
        # Call TEI/Infinity rerank API with retries
        for attempt in range(self.max_retries):
            try:
                inference_start = time.time()
                
                # TEI and Infinity use compatible API formats
                response = requests.post(
                    f"{self.base_url}/rerank",
                    json={
                        "query": query,
                        "texts": texts,  # TEI uses "texts"
                        "truncate": True,
                        "return_text": False  # We already have them
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                results = response.json()
                
                inference_time = (time.time() - inference_start) * 1000
                logger.debug(f"⚡ Inference completed in {inference_time:.2f}ms")
                logger.debug(f"   Throughput: {len(texts) / (inference_time / 1000):.1f} docs/sec")
                
                # Parse results: [{"index": 0, "score": 0.95}, ...]
                scored_items = []
                for item in results:
                    idx = item["index"]
                    score = item["score"]
                    if idx < len(valid_candidates):
                        scored_items.append((score, valid_candidates[idx]))
                
                # Sort by score (descending)
                scored_items.sort(key=lambda x: x[0], reverse=True)
                
                # Apply score threshold
                if score_threshold is not None:
                    before_filter = len(scored_items)
                    scored_items = [(s, c) for s, c in scored_items if s >= score_threshold]
                    logger.debug(f"🔽 Score threshold {score_threshold}: {before_filter} → {len(scored_items)}")
                
                # Apply top_k
                if top_k is not None and len(scored_items) > top_k:
                    scored_items = scored_items[:top_k]
                
                scores = [s for s, _ in scored_items]
                reranked = [c for _, c in scored_items]
                
                processing_time = (time.time() - start_time) * 1000
                
                logger.info(f"✅ Reranking completed in {processing_time:.2f}ms")
                if scores:
                    logger.info(f"   Results: {len(reranked)}/{len(candidates)} candidates")
                    logger.info(f"   Scores: top={scores[0]:.4f}, bottom={scores[-1]:.4f}")
                
                return RerankResult(
                    query=query,
                    candidates=candidates,
                    scores=scores if return_scores else [],
                    reranked_candidates=reranked,
                    processing_time_ms=processing_time,
                    model_name=self.model_name,
                    method="infinity"
                )
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏱️  Reranking timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Reranking request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
            except Exception as e:
                logger.error(f"❌ Unexpected error during reranking: {e}")
                break
        
        # Fallback: return original order if all retries failed
        logger.warning("⚠️  All reranking attempts failed, returning original order")
        processing_time = (time.time() - start_time) * 1000
        fallback_candidates = candidates[:top_k] if top_k else candidates
        
        return RerankResult(
            query=query,
            candidates=candidates,
            scores=[],
            reranked_candidates=fallback_candidates,
            processing_time_ms=processing_time,
            model_name=f"{self.model_name}_fallback",
            method="infinity_fallback"
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
                    'rerank_score': float(score) if score else 0.0,
                    'rerank_rank': i + 1,
                    'rerank_model': result.model_name,
                    'rerank_method': result.method,
                    'rerank_processing_time_ms': result.processing_time_ms
                })
            
            reranked_with_metadata.append(enhanced_candidate)
        
        return reranked_with_metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker.
        
        Returns:
            Dictionary with reranker information
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            is_available = response.status_code == 200
        except:
            is_available = False
        
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "service_available": is_available,
            "method": "infinity",
            "requires_torch": False,
            "inference_location": "remote_container"
        }


# Global instance for easy import
_reranker_instance: Optional[InfinityReranker] = None


def get_infinity_reranker() -> InfinityReranker:
    """
    Get or create global Infinity reranker instance.
    
    Environment variables:
        RERANKER_URL: Infinity service URL (default: http://reranker:7997)
        RERANKER_MODEL: Model name for logging (default: BAAI/bge-reranker-v2-m3)
        RERANKER_TIMEOUT: Request timeout in seconds (default: 30)
    
    Returns:
        InfinityReranker instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        base_url = os.getenv("RERANKER_URL", "http://reranker:7997")
        model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        timeout = int(os.getenv("RERANKER_TIMEOUT", "30"))
        
        logger.info("🚀 Initializing global Infinity reranker")
        
        _reranker_instance = InfinityReranker(
            base_url=base_url,
            model_name=model_name,
            timeout=timeout
        )
        
        info = _reranker_instance.get_model_info()
        logger.info(f"📋 Reranker info: {info}")
    
    return _reranker_instance


def rerank_candidates_infinity(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for reranking candidates using Infinity.
    
    Args:
        query: User query string
        candidates: List of candidate documents
        top_k: Number of top results to return
        
    Returns:
        List of reranked candidates with metadata
    """
    reranker = get_infinity_reranker()
    return reranker.rerank_with_metadata(query, candidates, top_k)


def is_reranking_available() -> bool:
    """Check if Infinity reranking service is available."""
    try:
        reranker = get_infinity_reranker()
        return reranker.get_model_info()["service_available"]
    except:
        return False


def get_reranking_info() -> Dict[str, Any]:
    """
    Get detailed information about reranking capabilities.
    
    Returns:
        Dictionary with reranking system information
    """
    reranker = get_infinity_reranker()
    info = reranker.get_model_info()
    
    info.update({
        "available_models": [
            "BAAI/bge-reranker-v2-m3",  # Multilingual, recommended
            "BAAI/bge-reranker-large",  # Higher quality
            "BAAI/bge-reranker-base",   # Faster
            "mixedbread-ai/mxbai-rerank-large-v1",  # Top quality
            "cross-encoder/ms-marco-mMiniLM-L-12-v2",  # Classic
        ],
        "recommended_model": "BAAI/bge-reranker-v2-m3",
        "status": "ready" if info["service_available"] else "unavailable",
        "notes": "Model can be changed in docker-compose.yml without code changes"
    })
    
    return info

