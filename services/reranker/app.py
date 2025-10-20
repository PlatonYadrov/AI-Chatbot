"""Simple Reranker HTTP Service using sentence-transformers."""

import os
import time
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model
model = None


class RerankRequest(BaseModel):
    """Rerank request model."""
    query: str
    texts: List[str]
    truncate: bool = True
    return_text: bool = False


class RerankResponse(BaseModel):
    """Rerank response model."""
    index: int
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    device = os.getenv("RERANKER_DEVICE", "cuda")
    cache_dir = os.getenv("HF_HOME", "/app/.cache")
    
    logger.info(f"🚀 Loading reranker model: {model_name}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Cache: {cache_dir}")
    
    try:
        start_time = time.time()
        model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
            trust_remote_code=False
        )
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("🔄 Shutting down reranker service")


app = FastAPI(
    title="Reranker Service",
    description="Cross-encoder reranking service using sentence-transformers",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")}


@app.post("/rerank", response_model=List[RerankResponse])
def rerank(request: RerankRequest):
    """
    Rerank texts based on query relevance.
    
    Args:
        request: RerankRequest with query and texts
        
    Returns:
        List of RerankResponse sorted by relevance score (descending)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        return []
    
    logger.info(f"🔍 Reranking {len(request.texts)} texts")
    
    try:
        start_time = time.time()
        
        # Create query-text pairs
        pairs = [[request.query, text] for text in request.texts]
        
        # Get relevance scores
        scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
        
        # Create response with index and score
        results = [
            RerankResponse(index=i, score=float(score))
            for i, score in enumerate(scores)
        ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        inference_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Reranking completed in {inference_time:.2f}ms")
        logger.info(f"   Top score: {results[0].score:.4f}, Bottom score: {results[-1].score:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "reranker",
        "model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "status": "running",
        "endpoints": {
            "health": "/health",
            "rerank": "/rerank (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8081"))
    logger.info(f"🚀 Starting reranker service on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

