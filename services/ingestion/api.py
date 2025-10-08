from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from processors.normalizer import normalize_text
from processors.chunker import chunk_text
from processors.deduplicator import deduplicate_chunks
from parsers.pdf_parser import PdfParser
from parsers.docx_parser import DocxParser
from parsers.pptx_parser import PptxParser
from parsers.xlsx_parser import XlsxParser
from parsers.ebook_parser import EbookParser
from parsers.scorm_parser import ScormParser
from parsers.base import RawBlock


OCR_URL = os.getenv("OCR_URL", "http://ocr:9000/ocr")
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
        resp = requests.post(f"{self.base_url}/embed", json={"input": texts}, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [item.get("embedding", []) for item in data]


embeddings = TEIEmbeddings()
client = QdrantClient(url=QDRANT_URL)

def _ensure_collection():
    try:
        client.get_collection(QDRANT_COLLECTION)
    except Exception:
        # assume 768 dims for e5-base; adjust if using BGE-M3 (1024)
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=len(embeddings.embed_query("dim")), distance=Distance.COSINE),
        )

app = FastAPI(title="Ingestion Service")


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
    try:
        content = await file.read()
        filename = (file.filename or "document").strip()
        suffix = (filename.split(".")[-1] if "." in filename else "").lower()

        # 1) Persist upload to a temporary path for parsers that require a file path
        tmp_dir = Path(os.getenv("DATA_DIR", "/data")) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename
        tmp_path.write_bytes(content)

        # 2) Choose parser by file type
        parser_blocks: list[RawBlock] = []
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        if suffix == "pdf":
            parser_blocks = list(PdfParser().parse(str(tmp_path), doc_id=doc_id))
        elif suffix == "docx":
            parser_blocks = list(DocxParser().parse(str(tmp_path), doc_id=doc_id))
        elif suffix == "pptx":
            parser_blocks = list(PptxParser().parse(str(tmp_path), doc_id=doc_id))
        elif suffix == "xlsx":
            parser_blocks = list(XlsxParser().parse(str(tmp_path), doc_id=doc_id))
        elif suffix in ("epub", "fb2"):
            parser_blocks = list(EbookParser().parse(str(tmp_path), doc_id=doc_id))
        elif suffix == "zip":
            parser_blocks = list(ScormParser().parse(str(tmp_path), doc_id=doc_id))
        else:
            # Fallback: try OCR for images; else treat as UTF-8 text
            if (file.content_type or "").startswith("image/") or suffix in {"png","jpg","jpeg","tif","tiff"}:
                files = {"file": (filename, content, file.content_type or "application/octet-stream")}
                ocr_resp = requests.post(OCR_URL, files=files, timeout=180)
                ocr_resp.raise_for_status()
                raw_text = ocr_resp.json().get("text", "")
            else:
                raw_text = content.decode("utf-8", errors="ignore")
            if raw_text.strip():
                parser_blocks = [RawBlock(text=raw_text, meta={"doc_id": doc_id, "type": "raw", "path": str(tmp_path)})]

        if not parser_blocks:
            raise HTTPException(status_code=400, detail="Parser produced no content")

        # 3) Normalize and chunk each block with rich metadata
        doc_metadata = {
            "doc_id": doc_id,
            "source_uri": filename,
            "source_type": suffix or "upload",
        }
        chunk_dicts: list[dict[str, Any]] = []
        for block in parser_blocks:
            normalized = normalize_text(block.text)
            if not normalized:
                continue
            # merge block-level metadata into document metadata for chunker
            merged_meta: dict[str, Any] = {**doc_metadata, **(block.meta or {})}
            for chunk in chunk_text(normalized, doc_metadata=merged_meta, lang="en"):
                chunk_dicts.append(chunk.to_dict())

        if not chunk_dicts:
            raise HTTPException(status_code=400, detail="No chunks produced")

        # 4) Deduplicate
        unique_chunks = deduplicate_chunks(chunk_dicts, threshold=0.85)

        # 5) Upsert into Qdrant with metadata
        _ensure_collection()
        texts = [c["text"] for c in unique_chunks]
        metadatas = [c["metadata"] for c in unique_chunks]

        Qdrant.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            url=QDRANT_URL,
            prefer_grpc=False,
            collection_name=QDRANT_COLLECTION,
        )

        return JSONResponse({
            "status": "ok",
            "doc_id": doc_id,
            "chunks_total": len(chunk_dicts),
            "chunks_unique": len(unique_chunks),
        })
    finally:
        await file.close()


