from __future__ import annotations

import hashlib
import uuid
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
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from processors.chunker import chunk_text
from processors.docling_chunker import DoclingChunker, DoclingChunk
from processors.deduplicator import deduplicate_chunks
from parsers.docling_parser import DoclingParser
from parsers.base import RawBlock


OCR_URL = os.getenv("OCR_URL", "http://ocr:9000/ocr")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

# Docling Chunking Configuration
USE_DOCLING_CHUNKING = os.getenv("USE_DOCLING_CHUNKING", "false").lower() == "true"
MAX_TOKENS = int(os.getenv("CHUNKING_MAX_TOKENS", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNKING_OVERLAP", "75"))


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
            payload = resp.json()
            # TEI may return one of:
            # 1) [{...vector...}, {...}] or [[...], [...]]
            # 2) {"data": [{"embedding": [...]}, ...]}
            # 3) {"embeddings": [[...], [...]]}
            if isinstance(payload, list):
                embeddings = [p.get("embedding", []) if isinstance(p, dict) else p for p in payload]
            elif isinstance(payload, dict):
                if "data" in payload:
                    embeddings = [item.get("embedding", []) for item in payload.get("data", [])]
                elif "embeddings" in payload:
                    embeddings = payload.get("embeddings", [])
                elif "embedding" in payload:
                    embeddings = [payload.get("embedding", [])]
                else:
                    embeddings = []
            else:
                embeddings = []
            LOGGER.info(
                "embeddings_ok endpoint=%s count=%s elapsed_ms=%s",
                f"{self.base_url}/embed",
                len(texts),
                int((time.time() - start_ts) * 1000),
            )
            return embeddings
        except Exception:
            LOGGER.exception(
                "embeddings_failed",
                extra={"endpoint": f"{self.base_url}/embed", "count": len(texts)},
            )
            raise


embeddings = TEIEmbeddings()
client = QdrantClient(url=QDRANT_URL)

# Initialize Docling Chunker (lazy, only if enabled)
docling_chunker = None
if USE_DOCLING_CHUNKING:
    # Auto-detect tokenizer from embeddings model or use explicit config
    tokenizer = os.getenv("CHUNKING_TOKENIZER")
    if not tokenizer:
        embeddings_model = os.getenv("EMBEDDINGS_MODEL", "intfloat/e5-base")
        if "e5" in embeddings_model or "bert" in embeddings_model.lower() or "bge" in embeddings_model.lower():
            tokenizer = "bert-base-uncased"  # BERT-based: e5, BGE, BERT
        elif "gpt" in embeddings_model.lower():
            tokenizer = "cl100k_base"  # GPT-based
        else:
            tokenizer = "bert-base-uncased"  # Default
    
    docling_chunker = DoclingChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        overlap=CHUNK_OVERLAP,
        merge_peers=True,
        include_metadata=True,
    )
    LOGGER.info(
        "docling_chunking_enabled",
        extra={
            "max_tokens": MAX_TOKENS,
            "overlap": CHUNK_OVERLAP,
            "tokenizer": tokenizer,
            "embeddings_model": os.getenv("EMBEDDINGS_MODEL", "unknown"),
        },
    )

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

        # 2) Parse document using Docling (unified parser for all formats)
        parser_blocks: list[RawBlock] = []
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        
        # Docling supports: PDF, DOCX, PPTX, XLSX, images (PNG, JPG, TIFF), HTML, and more
        supported_formats = {
            "pdf", "docx", "pptx", "xlsx", "xls", "doc", "ppt",
            "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif",
            "html", "htm", "md", "txt"
        }
        
        try:
            if suffix in supported_formats or content_type.startswith("image/"):
                LOGGER.info("select_parser", extra={"parser": "docling", "path": str(tmp_path), "suffix": suffix})
                
                # Configure parser from environment
                parser = DoclingParser(
                    extract_tables=os.getenv("DOCLING_EXTRACT_TABLES", "true").lower() == "true",
                    extract_images=True,
                    ocr_enabled=os.getenv("DOCLING_OCR_ENABLED", "true").lower() == "true",
                    preserve_structure=True,
                    offline_mode=os.getenv("DOCLING_OFFLINE_MODE", "true").lower() == "true",
                    cache_dir=os.getenv("DOCLING_CACHE_DIR"),
                    use_vlm=os.getenv("DOCLING_USE_VLM", "false").lower() == "true",  # VLM off by default
                )
                parser_blocks = list(parser.parse(str(tmp_path), doc_id=doc_id))
            else:
                # Fallback for unsupported formats: treat as plain text
                LOGGER.info("fallback_text_parser", extra={"suffix": suffix, "path": str(tmp_path)})
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

        # 3) Chunk with Docling HybridChunker (structure-aware) or traditional chunking
        doc_metadata = {
            "doc_id": doc_id,
            "source_uri": filename,
            "source_type": suffix or "upload",
        }
        chunk_dicts: list[dict[str, Any]] = []
        
        # Strategy 1: Docling HybridChunker (structure-aware, recommended)
        use_hybrid_chunking = USE_DOCLING_CHUNKING and docling_chunker and suffix in supported_formats
        if use_hybrid_chunking:
            try:
                # Get DoclingDocument for structure-aware chunking
                docling_doc = parser.parse_to_document(str(tmp_path), doc_id=doc_id)
                
                # Use HybridChunker to create structure-aware chunks
                docling_chunks = docling_chunker.chunk_document(
                    docling_doc,
                    doc_metadata=doc_metadata,
                )
                
                for chunk in docling_chunks:
                    chunk_dicts.append(chunk.to_dict())
                
                LOGGER.info(
                    "docling_chunking_complete",
                    extra={"doc_id": doc_id, "chunks": len(chunk_dicts), "strategy": "hybrid"},
                )
            except Exception as e:
                # Fallback to traditional chunking on error
                LOGGER.warning(
                    "docling_chunking_fallback",
                    extra={"doc_id": doc_id, "error": str(e)},
                )
                use_hybrid_chunking = False  # Disable for this request
        
        # Strategy 2: Traditional chunking (fallback or default)
        if not chunk_dicts:
            for block in parser_blocks:
                # Docling provides pre-cleaned text, no additional normalization needed
                if not block.text or not block.text.strip():
                    continue
                merged_meta: dict[str, Any] = {**doc_metadata, **(block.meta or {})}
                for chunk in chunk_text(block.text, doc_metadata=merged_meta, lang="en"):
                    chunk_dicts.append(chunk.to_dict())
            
            LOGGER.info(
                "traditional_chunking_complete",
                extra={"doc_id": doc_id, "chunks": len(chunk_dicts), "strategy": "traditional"},
            )

        if not chunk_dicts:
            LOGGER.warning("no_chunks doc_id=%s filename=%s", doc_id, filename)
            raise HTTPException(status_code=400, detail="No chunks produced")

        # 4) Deduplicate
        # unique_chunks = deduplicate_chunks(chunk_dicts, threshold=0.99)
        unique_chunks = chunk_dicts
        LOGGER.info(
            "chunks_ready",
            extra={
                "doc_id": doc_id,
                "chunks_total": len(chunk_dicts),
                "chunks_unique": len(unique_chunks),
            },
        )

        # 5) Upsert into Qdrant with deterministic IDs (chunk_id)
        _ensure_collection()
        
        # Truncate texts to max model context length
        # BGE-M3: 8192 tokens ≈ 32768 chars
        # e5-base: 512 tokens ≈ 2048 chars
        embeddings_model = os.getenv("EMBEDDINGS_MODEL", "intfloat/e5-base")
        if "bge-m3" in embeddings_model.lower():
            MAX_CHARS = 32768  # BGE-M3 supports 8192 tokens
        elif "mistral" in embeddings_model.lower():
            MAX_CHARS = 131072  # e5-mistral supports 32768 tokens
        else:
            MAX_CHARS = 2048  # Default: e5-base (512 tokens)
        texts = []
        for c in unique_chunks:
            text = c["text"]
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]
                LOGGER.debug(f"truncated_chunk original={len(c['text'])} truncated={len(text)}")
            texts.append(text)
        
        metadatas = [c["metadata"] for c in unique_chunks]

        try:
            start_qdrant = time.time()
            
            # Batch embeddings to avoid exceeding TEI max_client_batch_size
            # TEI on CPU has max_client_batch_size=32
            batch_size = 32
            vectors = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = embeddings.embed_documents(batch)
                vectors.extend(batch_vectors)
                LOGGER.debug(f"embedded_batch batch={i//batch_size + 1} size={len(batch)}")
            
            points = []
            for text_value, vec, meta in zip(texts, vectors, metadatas):
                # Use deterministic UUID based on chunk identity to satisfy Qdrant ID constraints
                raw_id = meta.get("chunk_id") or "_".join(
                    [
                        str(meta.get("doc_id", "")),
                        str(meta.get("language", "")),
                        str(meta.get("type", "")),
                        str(meta.get("path", "")),
                        str(meta.get("source_uri", "")),
                        str(meta.get("chunk_index", "0")),
                    ]
                )
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
                payload = {**meta}
                # persist chunk text for downstream inspection
                payload.setdefault("text", text_value)
                points.append(PointStruct(id=pid, vector=vec, payload=payload))
            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
            LOGGER.info(
                "qdrant_upsert_ok",
                extra={
                    "collection": QDRANT_COLLECTION,
                    "count": len(points),
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


