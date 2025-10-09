from __future__ import annotations

import hashlib
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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


LOGGER = logging.getLogger("ingestion.api")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


class TEIEmbeddings(Embeddings):
    def __init__(self, base_url: str = os.getenv("EMBEDDINGS_BASE_URL", "http://embeddings:80")):
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        start_ts = time.time()
        try:
            resp = requests.post(f"{self.base_url}/embed", json={"inputs": texts}, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            embeddings = [item.get("embedding", []) for item in data]
            LOGGER.info(
                "embeddings_ok",
                extra={
                    "endpoint": f"{self.base_url}/embed",
                    "count": len(texts),
                    "elapsed_ms": int((time.time() - start_ts) * 1000),
                },
            )
            return embeddings
        except Exception as exc:
            LOGGER.exception(
                "embeddings_failed",
                extra={"endpoint": f"{self.base_url}/embed", "count": len(texts)},
            )
            raise


embeddings = TEIEmbeddings()
client = QdrantClient(url=QDRANT_URL)

def _ensure_collection():
    try:
        client.get_collection(QDRANT_COLLECTION)
        return
    except Exception as exc:
        LOGGER.info("qdrant_collection_missing", extra={"collection": QDRANT_COLLECTION})
    try:
        size = len(embeddings.embed_query("dim"))
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        LOGGER.info("qdrant_collection_ready", extra={"collection": QDRANT_COLLECTION, "size": size})
    except Exception:
        LOGGER.exception("qdrant_collection_failed", extra={"collection": QDRANT_COLLECTION})
        raise

app = FastAPI(title="Ingestion Service")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_ts = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = int((time.time() - start_ts) * 1000)
        LOGGER.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": elapsed_ms,
                "client": getattr(request.client, "host", None),
            },
        )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
    try:
        content = await file.read()
        filename = (file.filename or "document").strip()
        suffix = (filename.split(".")[-1] if "." in filename else "").lower()
        content_type = file.content_type or ""
        LOGGER.info(
            "upload_received filename=%s suffix=%s content_type=%s size=%s",
            filename,
            suffix,
            content_type,
            len(content),
        )

        # 1) Persist upload to a temporary path for parsers that require a file path
        tmp_dir = Path(os.getenv("DATA_DIR", "/data")) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename
        tmp_path.write_bytes(content)

        # 2) Choose parser by file type
        parser_blocks: list[RawBlock] = []
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        try:
            if suffix == "pdf":
                LOGGER.info("select_parser", extra={"parser": "pdf", "path": str(tmp_path)})
                parser_blocks = list(PdfParser().parse(str(tmp_path), doc_id=doc_id))
            elif suffix == "docx":
                LOGGER.info("select_parser", extra={"parser": "docx", "path": str(tmp_path)})
                parser_blocks = list(DocxParser().parse(str(tmp_path), doc_id=doc_id))
            elif suffix == "pptx":
                LOGGER.info("select_parser", extra={"parser": "pptx", "path": str(tmp_path)})
                parser_blocks = list(PptxParser().parse(str(tmp_path), doc_id=doc_id))
            elif suffix == "xlsx":
                LOGGER.info("select_parser", extra={"parser": "xlsx", "path": str(tmp_path)})
                parser_blocks = list(XlsxParser().parse(str(tmp_path), doc_id=doc_id))
            elif suffix in ("epub", "fb2"):
                LOGGER.info("select_parser", extra={"parser": "ebook", "path": str(tmp_path)})
                parser_blocks = list(EbookParser().parse(str(tmp_path), doc_id=doc_id))
            elif suffix == "zip":
                LOGGER.info("select_parser", extra={"parser": "scorm", "path": str(tmp_path)})
                parser_blocks = list(ScormParser().parse(str(tmp_path), doc_id=doc_id))
            else:
                # Fallback: try OCR for images; else treat as UTF-8 text
                if content_type.startswith("image/") or suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
                    files = {"file": (filename, content, content_type or "application/octet-stream")}
                    start_ocr = time.time()
                    try:
                        ocr_resp = requests.post(OCR_URL, files=files, timeout=180)
                        ocr_resp.raise_for_status()
                        raw_text = ocr_resp.json().get("text", "")
                        LOGGER.info(
                            "ocr_ok",
                            extra={"endpoint": OCR_URL, "elapsed_ms": int((time.time() - start_ocr) * 1000)},
                        )
                    except Exception:
                        LOGGER.exception("ocr_failed", extra={"endpoint": OCR_URL})
                        raw_text = ""
                else:
                    raw_text = content.decode("utf-8", errors="ignore")
                if raw_text.strip():
                    parser_blocks = [
                        RawBlock(text=raw_text, meta={"doc_id": doc_id, "type": "raw", "path": str(tmp_path)})
                    ]
        except HTTPException:
            raise
        except Exception:
            LOGGER.exception("parse_failed", extra={"path": str(tmp_path), "suffix": suffix})
            raise HTTPException(status_code=500, detail="Parsing failed")

        if not parser_blocks:
            LOGGER.warning("no_blocks filename=%s suffix=%s", filename, suffix)
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
            merged_meta: dict[str, Any] = {**doc_metadata, **(block.meta or {})}
            for chunk in chunk_text(normalized, doc_metadata=merged_meta, lang="en"):
                chunk_dicts.append(chunk.to_dict())

        if not chunk_dicts:
            LOGGER.warning("no_chunks doc_id=%s filename=%s", doc_id, filename)
            raise HTTPException(status_code=400, detail="No chunks produced")

        # 4) Deduplicate
        unique_chunks = deduplicate_chunks(chunk_dicts, threshold=0.85)
        LOGGER.info(
            "chunks_ready",
            extra={
                "doc_id": doc_id,
                "chunks_total": len(chunk_dicts),
                "chunks_unique": len(unique_chunks),
            },
        )

        # 5) Upsert into Qdrant with metadata
        _ensure_collection()
        texts = [c["text"] for c in unique_chunks]
        metadatas = [c["metadata"] for c in unique_chunks]

        try:
            start_qdrant = time.time()
            Qdrant.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                url=QDRANT_URL,
                prefer_grpc=False,
                collection_name=QDRANT_COLLECTION,
            )
            LOGGER.info(
                "qdrant_upsert_ok",
                extra={
                    "collection": QDRANT_COLLECTION,
                    "count": len(texts),
                    "elapsed_ms": int((time.time() - start_qdrant) * 1000),
                },
            )
        except Exception:
            LOGGER.exception(
                "qdrant_upsert_failed",
                extra={"collection": QDRANT_COLLECTION, "count": len(texts)},
            )
            raise HTTPException(status_code=502, detail="Vector store unavailable")

        return JSONResponse(
            {
                "status": "ok",
                "doc_id": doc_id,
                "chunks_total": len(chunk_dicts),
                "chunks_unique": len(unique_chunks),
            }
        )
    finally:
        try:
            await file.close()
        except Exception:
            LOGGER.debug("file_close_failed")


