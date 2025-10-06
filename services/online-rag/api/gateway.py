from __future__ import annotations

from typing import Any, Dict, List

from fastapi import Depends, FastAPI

from .auth_middleware import UserContext, get_user_ctx
from .rate_limiter import limiter
from ..nlp.intent_classifier import classify
from ..nlp.entity_extractor import extract
from ..search.query_embedder import encode as encode_query
from ..search.hybrid_search import search as hybrid
from ..reranker.cross_encoder_service import rerank
from ..generation.prompt_builder import build as build_prompt
from ..generation.llm_service import generate
from ..generation.guardrails import validate


app = FastAPI(title="Highland AI Chat Gateway")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> Dict[str, str]:
    return {"status": "ready"}


@app.post("/search")
def search(body: Dict[str, Any], user: UserContext = Depends(get_user_ctx), _rl: None = Depends(limiter(100))) -> Dict[str, Any]:
    query = body.get("query", "")
    filters = body.get("filters", {})
    intent = classify(query)
    entities = extract(query)
    qvec = encode_query(query)
    candidates = hybrid(query, qvec, filters)
    top = rerank(query, candidates, top_k=10)
    return {"intent": intent, "entities": entities, "results": top}


@app.post("/chat")
def chat(body: Dict[str, Any], user: UserContext = Depends(get_user_ctx), _rl: None = Depends(limiter(60))) -> Dict[str, Any]:
    query = body.get("query", "")
    filters = body.get("filters", {})
    qvec = encode_query(query)
    candidates = hybrid(query, qvec, filters)
    top = rerank(query, candidates, top_k=10)
    prompt = build_prompt(query, top)
    out = generate(query, top, mode="rag")
    check = validate(out.get("text", ""), top)
    return {"text": out["text"], "usage": out["usage"], "guard": check, "citations": top[:5]}


__all__ = ["app"]

