# Quick Start: Docling Chunking

## TL;DR

Enable structure-aware chunking in 2 lines:

```python
pipeline = LocalIngestionPipeline(
    embedder=EmbedderService(),
    use_docling_chunking=True,  # ← Enable HybridChunker
)
```

## Why?

**Traditional chunking** splits text at arbitrary positions → breaks context

**Docling HybridChunker** respects document structure → preserves context ✓

## Installation

```bash
pip install docling>=2.0.0 docling-core[chunking]>=2.0.0
```

Already in `requirements.txt`! Just run:

```bash
cd services/ingestion
pip install -r requirements.txt
```

## Usage

### 1. Basic Usage (Recommended)

```python
from services.ingestion.pipline import LocalIngestionPipeline
from services.ingestion.embedder_service import EmbedderService

pipeline = LocalIngestionPipeline(
    embedder=EmbedderService(),
    use_docling_chunking=True,  # Enable HybridChunker
    max_tokens=512,              # Chunk size in tokens
)

# Process documents
records = pipeline.ingest_path("document.pdf")
```

### 2. Advanced Configuration

```python
pipeline = LocalIngestionPipeline(
    embedder=EmbedderService(),
    use_docling=True,             # Use Docling parser
    use_docling_chunking=True,    # Enable HybridChunker
    max_tokens=512,               # Max tokens per chunk
    chunk_overlap=75,             # Token overlap
)
```

### 3. Direct Chunker Usage

```python
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.docling_chunker import DoclingChunker

# Parse document
parser = DoclingParser()
doc = parser.parse_to_document("document.pdf")

# Chunk document
chunker = DoclingChunker(max_tokens=512, overlap=75)
chunks = chunker.chunk_document(doc)

# Use chunks
for chunk in chunks:
    print(f"Page {chunk.meta['page']}: {chunk.text[:100]}...")
```

## Test It

```bash
# Install dependencies
pip install -r services/ingestion/requirements.txt

# Test chunking
python test_docling_chunking.py

# Or with your own PDF
python test_docling_chunking.py path/to/your.pdf
```

## What You Get

Each chunk includes:

```python
{
    "text": "chunk text...",
    "metadata": {
        "chunk_index": 0,
        "page": 5,
        "headings": ["Chapter 1", "Section 1.2"],
        "doc_id": "document_123",
        "chunking_strategy": "docling_hybrid"
    }
}
```

## Comparison

| Feature | Traditional | Docling HybridChunker |
|---------|------------|----------------------|
| Respects headings | ❌ | ✅ |
| Preserves tables | ❌ | ✅ |
| Token-accurate | ⚠️ | ✅ |
| Metadata | Basic | Rich |

## Full Documentation

See [`DOCLING_CHUNKING.md`](./DOCLING_CHUNKING.md) for:
- Detailed architecture
- All configuration options
- Advanced examples
- Performance tips
- Troubleshooting

## Migration from Traditional Chunking

**Before:**
```python
pipeline = LocalIngestionPipeline(
    embedder=embedder,
    chunk_size=1200,           # Characters
    chunk_strategy="token_aware",
)
```

**After:**
```python
pipeline = LocalIngestionPipeline(
    embedder=embedder,
    use_docling_chunking=True,  # Enable HybridChunker
    max_tokens=512,             # Tokens (more precise)
)
```

That's it! Your pipeline now uses structure-aware chunking. 🚀

