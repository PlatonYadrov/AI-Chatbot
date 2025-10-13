# Docling Chunking Integration

## Overview

Docling's **HybridChunker** provides advanced, structure-aware document chunking that's superior to naive text splitting. It uses document layout analysis and semantic understanding to determine optimal chunk boundaries.

## Why Use Docling Chunking?

### Traditional Text Chunking Problems
- ❌ Splits at arbitrary character/token counts
- ❌ Ignores document structure (headings, sections)
- ❌ May break sentences or paragraphs mid-way
- ❌ Loses semantic coherence
- ❌ Can split tables, lists, or code blocks

### Docling HybridChunker Benefits
- ✅ Respects document structure (headings, sections, paragraphs)
- ✅ Maintains semantic coherence
- ✅ Token-aware (uses real tokenizer)
- ✅ Preserves tables, lists, and code blocks
- ✅ Includes document hierarchy in metadata
- ✅ Better context for RAG systems

## Architecture

```
Document (PDF/DOCX/etc.)
    ↓
DoclingParser (layout analysis)
    ↓
DoclingDocument (structured representation)
    ↓
HybridChunker (structure-aware splitting)
    ↓
Chunks with metadata
```

## Installation

The necessary dependencies are already included in `requirements.txt`:

```bash
docling>=2.0.0
docling-core[chunking]>=2.0.0
```

Install them:

```bash
pip install -r services/ingestion/requirements.txt
```

## Usage

### Option 1: Use in Pipeline (Recommended)

Enable Docling chunking in the ingestion pipeline:

```python
from services.ingestion.pipline import LocalIngestionPipeline
from services.ingestion.embedder_service import EmbedderService

# Initialize with Docling chunking enabled
pipeline = LocalIngestionPipeline(
    embedder=EmbedderService(),
    use_docling=True,                # Use Docling parser
    use_docling_chunking=True,       # Enable HybridChunker
    max_tokens=512,                   # Max tokens per chunk
    chunk_overlap=75,                 # Token overlap between chunks
)

# Ingest a document
records = pipeline.ingest_path("document.pdf")

# Each record has:
# - text: chunk text
# - vector: embedding
# - payload: metadata (headings, page, structure, etc.)
for record in records:
    print(f"Chunk {record.id}:")
    print(f"  Text: {record.text[:100]}...")
    print(f"  Metadata: {record.payload.get('headings', [])}")
```

### Option 2: Direct Usage

Use the chunker directly for custom workflows:

```python
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.docling_chunker import DoclingChunker

# Step 1: Parse document to DoclingDocument
parser = DoclingParser(
    extract_tables=True,
    ocr_enabled=True,
)
docling_doc = parser.parse_to_document("document.pdf")

# Step 2: Chunk with HybridChunker
chunker = DoclingChunker(
    max_tokens=512,
    overlap=75,
    merge_peers=True,       # Merge small adjacent chunks
    include_metadata=True,  # Include structural metadata
)

chunks = chunker.chunk_document(
    docling_doc,
    doc_metadata={
        "doc_id": "my_document",
        "source": "internal",
    }
)

# Use chunks
for chunk in chunks:
    print(f"Chunk {chunk.meta['chunk_index']}:")
    print(f"  Text: {chunk.text}")
    print(f"  Headings: {chunk.meta.get('headings', [])}")
    print(f"  Page: {chunk.meta.get('page', 'N/A')}")
```

### Option 3: Lazy Iterator (Memory Efficient)

For large documents, use lazy iteration:

```python
from services.ingestion.processors.docling_chunker import DoclingChunker

chunker = DoclingChunker(max_tokens=512)

# Yields chunks one by one (no need to load all in memory)
for chunk in chunker.chunk_document_lazy(docling_doc):
    process_chunk(chunk)  # Process immediately
```

## Configuration

### Pipeline Configuration

```python
pipeline = LocalIngestionPipeline(
    embedder=embedder,
    
    # Docling settings
    use_docling=True,                # Use Docling parser (recommended)
    use_docling_chunking=True,       # Enable HybridChunker
    
    # Chunking parameters
    max_tokens=512,                   # Max tokens per chunk (default: 512)
    chunk_overlap=75,                 # Token overlap (default: 75)
    
    # Traditional chunking (if use_docling_chunking=False)
    chunk_size=1200,                  # Characters (not used with Docling chunking)
    chunk_strategy="token_aware",     # Only if not using Docling chunking
)
```

### Chunker Parameters

```python
chunker = DoclingChunker(
    tokenizer="cl100k_base",    # Tokenizer name (GPT-3.5/4 default)
    max_tokens=512,              # Max tokens per chunk
    overlap=75,                  # Token overlap between chunks
    merge_peers=True,            # Merge small adjacent chunks for context
    include_metadata=True,       # Include document structure in metadata
)
```

### Tokenizer Options

Common tokenizers:
- `cl100k_base` - GPT-3.5-turbo, GPT-4 (default)
- `p50k_base` - GPT-3 (davinci, curie, etc.)
- `r50k_base` - GPT-3 (ada, babbage)

## Metadata

Each chunk includes rich metadata:

```python
chunk.meta = {
    "doc_id": "document_123",
    "path": "/path/to/document.pdf",
    "chunk_index": 5,
    "chunking_strategy": "docling_hybrid",
    
    # Document structure
    "headings": ["Chapter 1", "Section 1.2", "Subsection 1.2.3"],
    "page": 42,
    "pages": [42, 43],  # If chunk spans multiple pages
    
    # From Docling parser
    "label": "paragraph",  # or "title", "table", "list_item", etc.
    "level": 2,            # Heading level in document hierarchy
}
```

## Examples

### Example 1: Ingest PDF with Docling Chunking

```python
from services.ingestion.pipline import LocalIngestionPipeline
from services.ingestion.embedder_service import EmbedderService

pipeline = LocalIngestionPipeline(
    embedder=EmbedderService(),
    use_docling=True,
    use_docling_chunking=True,
    max_tokens=512,
)

# Process a document
records = pipeline.ingest_path("research_paper.pdf")

print(f"Created {len(records)} chunks")
for record in records[:3]:  # Show first 3 chunks
    print(f"\nChunk: {record.payload.get('chunk_index')}")
    print(f"Headings: {record.payload.get('headings', [])}")
    print(f"Page: {record.payload.get('page')}")
    print(f"Text preview: {record.text[:200]}...")
```

### Example 2: Compare Chunking Strategies

```python
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.docling_chunker import DoclingChunker
from services.ingestion.processors.chunker import chunk_text

parser = DoclingParser()
docling_doc = parser.parse_to_document("document.pdf")

# Strategy 1: Docling HybridChunker (structure-aware)
docling_chunker = DoclingChunker(max_tokens=512)
hybrid_chunks = docling_chunker.chunk_document(docling_doc)

print(f"HybridChunker: {len(hybrid_chunks)} chunks")
for chunk in hybrid_chunks[:2]:
    print(f"  - {chunk.text[:100]}... (page {chunk.meta.get('page')})")

# Strategy 2: Traditional text splitting
full_text = docling_doc.export_to_markdown()
text_chunks = chunk_text(full_text, chunk_size_tokens=512, strategy="token_aware")

print(f"\nTraditional: {len(text_chunks)} chunks")
for chunk in text_chunks[:2]:
    print(f"  - {chunk.text[:100]}...")
```

### Example 3: Batch Processing

```python
from pathlib import Path
from services.ingestion.pipline import LocalIngestionPipeline
from services.ingestion.embedder_service import EmbedderService

pipeline = LocalIngestionPipeline(
    embedder=EmbedderService(),
    use_docling_chunking=True,
    max_tokens=512,
)

# Process all documents in a directory
docs_dir = Path("documents/")
all_records = []

for doc_path in docs_dir.glob("*.pdf"):
    print(f"Processing {doc_path.name}...")
    records = pipeline.ingest_path(doc_path)
    all_records.extend(records)
    print(f"  → {len(records)} chunks")

print(f"\nTotal: {len(all_records)} chunks from {len(list(docs_dir.glob('*.pdf')))} documents")
```

## Performance Tips

1. **Use lazy iteration for large documents**:
   ```python
   for chunk in chunker.chunk_document_lazy(docling_doc):
       process_immediately(chunk)
   ```

2. **Adjust max_tokens based on your model**:
   - For embeddings: 512 tokens (default)
   - For LLM context: 1024-2048 tokens
   - For search: 256-512 tokens

3. **Enable merge_peers for better context**:
   ```python
   chunker = DoclingChunker(merge_peers=True)  # Merges small adjacent chunks
   ```

4. **Use appropriate overlap**:
   - Small overlap (50-75): Less redundancy, more unique chunks
   - Large overlap (100-150): Better context continuity, more redundancy

## Testing

Test the integration:

```bash
# Test Docling chunking
python -c "
from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.docling_chunker import DoclingChunker

parser = DoclingParser()
doc = parser.parse_to_document('tests/pdf/Knoleges.pdf')

chunker = DoclingChunker(max_tokens=512)
chunks = chunker.chunk_document(doc)

print(f'Created {len(chunks)} chunks')
for i, chunk in enumerate(chunks[:3]):
    print(f'Chunk {i}: {len(chunk.text)} chars, page {chunk.meta.get(\"page\")}')"
```

## Comparison: Traditional vs Docling Chunking

| Feature | Traditional | Docling HybridChunker |
|---------|------------|----------------------|
| Structure-aware | ❌ | ✅ |
| Token-accurate | ⚠️ (estimate) | ✅ (real tokenizer) |
| Respects headings | ❌ | ✅ |
| Preserves tables | ❌ | ✅ |
| Semantic coherence | ❌ | ✅ |
| Metadata richness | ⚠️ (basic) | ✅ (extensive) |
| Setup complexity | Simple | Medium |
| Performance | Fast | Slower (but smarter) |

## Troubleshooting

### Import Error: HybridChunker not found

```bash
pip install 'docling-core[chunking]'
```

### Chunking takes too long

- Use `chunk_document_lazy()` for large documents
- Reduce `max_tokens` to create more smaller chunks
- Disable `merge_peers` if not needed

### Chunks are too small/large

Adjust `max_tokens`:
```python
chunker = DoclingChunker(max_tokens=1024)  # Larger chunks
chunker = DoclingChunker(max_tokens=256)   # Smaller chunks
```

## References

- [Docling Documentation](https://github.com/docling-project/docling)
- [Docling Chunking Concepts](https://github.com/docling-project/docling/blob/main/docs/concepts/chunking.md)
- [HybridChunker API](https://github.com/docling-project/docling-core)

## Summary

Docling's HybridChunker provides intelligent, structure-aware document chunking that significantly improves RAG quality by:

1. **Preserving document structure** (headings, sections)
2. **Maintaining semantic coherence** within chunks
3. **Including rich metadata** (page numbers, hierarchy)
4. **Using real tokenizers** for accurate token counts

Enable it in your pipeline with `use_docling_chunking=True` for better search and retrieval results! 🚀

