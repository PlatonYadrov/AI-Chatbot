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

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qexc
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from processors.chunker import DocumentChunk, create_chunker, ChunkingConfig
from parsers.docling_parser import DoclingParser


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")

# Docling Chunking Configuration
MAX_TOKENS = int(os.getenv("CHUNKING_MAX_TOKENS", "512"))


LOGGER = logging.getLogger("ingestion.api")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


class TEIEmbeddings:
    def __init__(self, base_url: str = os.getenv("EMBEDDINGS_BASE_URL", "http://embeddings:80")):
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        start_ts = time.time()
        try:
            resp = requests.post(f"{self.base_url}/embed", json={"inputs": texts}, timeout=180)
            resp.raise_for_status()
            embeddings = resp.json()  # ожидаем в формате HF [[...], [...]]
            LOGGER.info("embeddings_ok endpoint=%s count=%s elapsed_ms=%s", f"{self.base_url}/embed", len(texts), int((time.time() - start_ts) * 1000))
            return embeddings
        except Exception:
            LOGGER.exception("embeddings_failed", extra={"endpoint": f"{self.base_url}/embed", "count": len(texts)})
            raise


embeddings = TEIEmbeddings()
qdrant_client = QdrantClient(url=QDRANT_URL)



def _ensure_collection():
    try:
        qdrant_client.get_collection(QDRANT_COLLECTION)
        LOGGER.debug(f"Collection '{QDRANT_COLLECTION}' already exists")
        return
    except qexc.ResponseHandlingException:
        LOGGER.info(f"Collection '{QDRANT_COLLECTION}' not found, creating...")

    try:
        size = len(embeddings.embed_query(""))
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        LOGGER.info(f"Collection '{QDRANT_COLLECTION}' created with vector size {size}")
    except Exception:
        LOGGER.exception("Failed to create Qdrant collection", extra={"collection": QDRANT_COLLECTION})
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
        if request.url.path not in ("/health", "/metrics"):
            LOGGER.info("http_request", extra={"method": request.method, "path": request.url.path, "status": getattr(response, "status_code", None), "elapsed_ms": elapsed_ms, "client": getattr(request.client, "host", None),})


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
    try:
        content = await file.read()
        filename = (file.filename or "document").strip()
        suffix = (filename.split(".")[-1] if "." in filename else "").lower()
        content_type = file.content_type or ""
        LOGGER.info("upload_received filename=%s suffix=%s content_type=%s size=%s", filename, suffix,  content_type,len(content))

        # 1) Persist upload to a temporary path for parsers that require a file path
        tmp_dir = Path(os.getenv("DATA_DIR", "/data")) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename
        tmp_path.write_bytes(content)

        # 2) Parse document using Docling (unified parser for all formats)
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        docling_doc = None
        md_text = None
        
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
                    use_vlm=os.getenv("DOCLING_USE_VLM", "false").lower() == "true",
                )
                LOGGER.debug("ocr_enabled=%s", os.getenv("DOCLING_OCR_ENABLED", "true").lower() == "true")
                md_text, docling_doc = parser.parse_to_document(str(tmp_path), doc_id=doc_id, markdown=True)
            else:
                # Fallback for unsupported formats: treat as plain text
                LOGGER.info("fallback_text_parser", extra={"suffix": suffix, "path": str(tmp_path)})
                md_text = content.decode("utf-8", errors="ignore")
        except HTTPException:
            raise
        except Exception:
            LOGGER.exception("parse_failed", extra={"path": str(tmp_path), "suffix": suffix})
            raise HTTPException(status_code=500, detail="Parsing failed")

        if not md_text or not md_text.strip():
            LOGGER.warning("no_content filename=%s suffix=%s", filename, suffix)
            raise HTTPException(status_code=400, detail="Parser produced no content")

        LOGGER.info(
            f"📄 Document parsed successfully: "
            f"size={len(md_text)} chars, "
            f"has_docling_doc={docling_doc is not None}, "
            f"ready_for_chunking=True"
        )

        # 3) Chunk with Docling HybridChunker (structure-aware) or fallback chunking
        doc_metadata = {
            "doc_id": doc_id,
            "source_uri": filename,
            "source_type": suffix or "upload",
        }
        chunk_dicts: list[dict[str, Any]] = []
        
        LOGGER.info(f"🔧 Chunking strategy: {'HybridChunker (structure-aware)' if docling_doc else 'SimpleChunker (fallback)'}")
        
        if docling_doc is not None:
            # Use Docling HybridChunker for structure-aware chunking
            try:
                LOGGER.info("using_docling_hybrid_chunker", extra={"doc_id": doc_id})
                
                # Create chunking configuration
                config = ChunkingConfig(
                    max_tokens=MAX_TOKENS,
                    chunk_size=1000,
                    chunk_overlap=200,
                    use_semantic_splitting=True,
                )
                
                # Initialize chunker
                chunker = create_chunker(config=config)
                
                # Chunk document (async method)
                docling_chunks: List[DocumentChunk] = await chunker.chunk_document(
                    content=md_text,
                    title=filename,
                    source=str(tmp_path),
                    metadata=doc_metadata,
                    docling_doc=docling_doc,
                )
                
                for i, chunk in enumerate(docling_chunks):
                    chunk_meta = {**chunk.metadata}
                    chunk_meta.setdefault("chunk_id", f"{doc_id}_chunk_{i}")
                    chunk_dict = {
                        "text": chunk.content,
                        "meta": chunk_meta,
                    }
                    chunk_dicts.append(chunk_dict)
                    
                    # Детальное логирование каждого чанка
                    preview_text = chunk.content[:100].replace('\n', ' ')
                    LOGGER.info(
                        f"[Chunk {i+1}/{len(docling_chunks)}] "
                        f"size={len(chunk.content)} chars, "
                        f"tokens={chunk_meta.get('token_count', 'N/A')}, "
                        f"preview='{preview_text}...'"
                    )
                
                LOGGER.info(
                    "docling_chunking_complete",
                    extra={"doc_id": doc_id, "chunks": len(chunk_dicts), "strategy": "docling_hybrid"},
                )
            except Exception as e:
                LOGGER.warning("docling_chunker_failed, falling back to simple chunking", extra={"error": str(e)})
                docling_doc = None
        
        # Fallback: use SimpleChunker if docling_doc is None or chunking failed
        if not chunk_dicts and md_text:
            try:
                LOGGER.info("using_simple_chunker_fallback", extra={"doc_id": doc_id})
                
                # Create simple chunker configuration
                config = ChunkingConfig(
                    max_tokens=MAX_TOKENS,
                    chunk_size=1000,
                    chunk_overlap=200,
                    use_semantic_splitting=False,  # Use SimpleChunker
                )
                
                # Initialize simple chunker
                chunker = create_chunker(config=config)
                
                # Chunk document (async method)
                simple_chunks: List[DocumentChunk] = await chunker.chunk_document(
                    content=md_text,
                    title=filename,
                    source=str(tmp_path),
                    metadata=doc_metadata,
                )
                
                for i, chunk in enumerate(simple_chunks):
                    chunk_meta = {**chunk.metadata}
                    chunk_meta.setdefault("chunk_id", f"{doc_id}_chunk_{i}")
                    chunk_dict = {
                        "text": chunk.content,
                        "meta": chunk_meta,
                    }
                    chunk_dicts.append(chunk_dict)
                    
                    # Детальное логирование каждого чанка
                    preview_text = chunk.content[:100].replace('\n', ' ')
                    LOGGER.info(
                        f"[SimpleChunk {i+1}/{len(simple_chunks)}] "
                        f"size={len(chunk.content)} chars, "
                        f"tokens={chunk_meta.get('token_count', 'N/A')}, "
                        f"preview='{preview_text}...'"
                    )
                
                LOGGER.info(
                    "simple_chunking_complete",
                    extra={"doc_id": doc_id, "chunks": len(chunk_dicts), "strategy": "simple_fallback"},
                )
            except Exception as e:
                LOGGER.exception("simple_chunker_also_failed", extra={"error": str(e)})

        if not chunk_dicts:
            LOGGER.warning("no_chunks doc_id=%s filename=%s", doc_id, filename)
            raise HTTPException(status_code=400, detail="No chunks produced")

        # Статистика по чанкам
        total_chars = sum(len(c["text"]) for c in chunk_dicts)
        avg_chars = total_chars // len(chunk_dicts) if chunk_dicts else 0
        min_chars = min(len(c["text"]) for c in chunk_dicts) if chunk_dicts else 0
        max_chars = max(len(c["text"]) for c in chunk_dicts) if chunk_dicts else 0
        
        LOGGER.info(
            f"✓ Chunking statistics: "
            f"total={len(chunk_dicts)} chunks, "
            f"avg_size={avg_chars} chars, "
            f"min={min_chars}, max={max_chars}, "
            f"total_text={total_chars} chars"
        )
        
        LOGGER.info("chunks_ready", extra={"doc_id": doc_id, "chunks_total": len(chunk_dicts)})

        # 4) Upsert into Qdrant with deterministic IDs (chunk_id)
        _ensure_collection()
        
        # TEI auto-truncate enabled - no manual truncation needed
        # Model will automatically truncate inputs to max context length
        texts = [c["text"] for c in chunk_dicts]
        metadatas = [c["meta"] for c in chunk_dicts]

        try:
            start_qdrant = time.time()
            
            LOGGER.info(f"🔢 Creating embeddings for {len(texts)} chunks...")
            
            # Batch embeddings to avoid exceeding TEI max_client_batch_size
            # TEI on CPU: reduce batch size for better timeout handling
            batch_size = 16  # Smaller batches on CPU (was 32)
            total_batches = (len(texts) + batch_size - 1) // batch_size
            vectors = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = embeddings.embed_documents(batch)
                vectors.extend(batch_vectors)
                batch_num = i//batch_size + 1
                LOGGER.info(f"  Batch {batch_num}/{total_batches}: embedded {len(batch)} chunks")
            
            embedding_time = time.time() - start_qdrant
            LOGGER.info(f"✓ Embeddings created in {embedding_time:.2f}s")
            
            LOGGER.info(f"💾 Preparing {len(vectors)} points for Qdrant...")
            
            points = []
            for text_value, vec, meta in zip(texts, vectors, metadatas):
                # Use deterministic UUID based on chunk identity to satisfy Qdrant ID constraints
                raw_id = meta.get("chunk_id") or "_".join(
                    [
                        str(meta.get("block_start", "")),
                        str(meta.get("block_end", "")),
                    ]
                )
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
                payload = {**meta}
                # persist chunk text for downstream inspection
                payload.setdefault("text", text_value)
                points.append(PointStruct(id=pid, vector=vec, payload=payload))
            
            LOGGER.info(f"⬆️  Upserting {len(points)} points to Qdrant collection '{QDRANT_COLLECTION}'...")
            upsert_start = time.time()
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
            upsert_time = time.time() - upsert_start
            total_time = time.time() - start_qdrant
            
            LOGGER.info(
                f"✅ Successfully uploaded to Qdrant: "
                f"{len(points)} vectors, "
                f"upsert_time={upsert_time:.2f}s, "
                f"total_time={total_time:.2f}s"
            )
            LOGGER.info(
                "qdrant_upsert_ok",
                extra={
                    "collection": QDRANT_COLLECTION,
                    "count": len(points),
                    "elapsed_ms": int(total_time * 1000),
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
                "chunks_count": len(chunk_dicts),
            }
        )
    finally:
        try:
            await file.close()
        except Exception:
            LOGGER.debug("file_close_failed")


