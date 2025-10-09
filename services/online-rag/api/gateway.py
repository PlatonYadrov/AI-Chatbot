from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
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
    answer = generate_response(prompt)
    return {"answer": answer, "sources": [getattr(d, 'metadata', {}) for d in docs]}
