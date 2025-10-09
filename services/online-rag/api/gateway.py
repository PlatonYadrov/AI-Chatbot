from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from generation.llm_service import generate_response


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")


class TEIEmbeddings(Embeddings):
    def __init__(self, base_url: str = os.getenv("EMBEDDINGS_BASE_URL", "http://embeddings:80")):
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(f"{self.base_url}/embed", json={"inputs": texts}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            return [p.get("embedding", []) if isinstance(p, dict) else p for p in payload]
        if isinstance(payload, dict):
            if "data" in payload:
                return [item.get("embedding", []) for item in payload.get("data", [])]
            if "embeddings" in payload:
                return payload.get("embeddings", [])
            if "embedding" in payload:
                return [payload.get("embedding", [])]
        return []


class QueryRequest(BaseModel):
    query: str
    k: int = 4
    include_vectors: bool = False
    vector_top_n: int = 10


app = FastAPI(title="RAG Gateway")


@app.post("/query")
def query_endpoint(req: QueryRequest):
    embeddings = TEIEmbeddings()
    vs = Qdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
        prefer_grpc=False,
        path=None,
    )
    docs = vs.similarity_search(req.query, k=req.k)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Answer the question using the context.\n\nContext:\n{context}\n\nQuestion: {req.query}"
    answer, thoughts = generate_response(prompt)

    sources = []
    for d in docs:
        meta = getattr(d, 'metadata', {})
        # try both common keys for chunk text
        if 'text' not in meta and hasattr(d, 'page_content'):
            meta = {**meta, 'text': getattr(d, 'page_content')}
        sources.append(meta)
    if req.include_vectors:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm
            client = QdrantClient(url=QDRANT_URL)
            # fetch points with vectors by chunk_id
            enriched = []
            for meta in sources:
                cid = meta.get("chunk_id")
                if not cid:
                    enriched.append(meta)
                    continue
                res = client.retrieve(collection_name=QDRANT_COLLECTION, ids=[cid], with_vectors=True)
                if res:
                    vec = res[0].vector
                    if not isinstance(vec, list):
                        # named vector
                        vec = list(vec.values())[0]
                    meta = {**meta, "vector_head": vec[: max(0, int(req.vector_top_n))], "vector_dim": len(vec)}
                enriched.append(meta)
            sources = enriched
        except Exception:
            # if enrichment fails, return sources as-is
            pass

    return {"answer": answer, "thoughts": thoughts, "sources": sources}
