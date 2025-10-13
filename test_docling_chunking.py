"""Test script for Docling HybridChunker integration.

Demonstrates structure-aware document chunking using Docling's HybridChunker.
"""

import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent))

from services.ingestion.parsers.docling_parser import DoclingParser
from services.ingestion.processors.docling_chunker import DoclingChunker


def test_docling_chunking(pdf_path: str = "tests/pdf/Knoleges.pdf"):
    """Test Docling chunking on a sample PDF."""
    
    print("=" * 80)
    print("Testing Docling HybridChunker")
    print("=" * 80)
    
    # Step 1: Parse document
    print(f"\n1. Parsing document: {pdf_path}")
    parser = DoclingParser(
        extract_tables=True,
        ocr_enabled=True,
        preserve_structure=True,
    )
    
    try:
        docling_doc = parser.parse_to_document(pdf_path)
        print(f"   ✓ Document parsed successfully")
        print(f"   ✓ Pages: {getattr(docling_doc, 'num_pages', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Parse failed: {e}")
        return
    
    # Step 2: Chunk with HybridChunker
    print(f"\n2. Chunking with HybridChunker")
    chunker = DoclingChunker(
        max_tokens=512,
        overlap=75,
        merge_peers=True,
        include_metadata=True,
    )
    
    try:
        chunks = chunker.chunk_document(
            docling_doc,
            doc_metadata={
                "doc_id": Path(pdf_path).stem,
                "source": "test",
            }
        )
        print(f"   ✓ Created {len(chunks)} chunks")
    except Exception as e:
        print(f"   ✗ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Display chunk details
    print(f"\n3. Chunk Analysis")
    print("-" * 80)
    
    total_chars = sum(len(chunk.text) for chunk in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0
    
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average chunk size: {avg_chars} characters")
    
    # Step 4: Show sample chunks
    print(f"\n4. Sample Chunks (first 3)")
    print("-" * 80)
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n   Chunk {i + 1}:")
        print(f"   ├─ Index: {chunk.meta.get('chunk_index')}")
        print(f"   ├─ Page: {chunk.meta.get('page', 'N/A')}")
        print(f"   ├─ Strategy: {chunk.meta.get('chunking_strategy')}")
        
        headings = chunk.meta.get('headings', [])
        if headings:
            print(f"   ├─ Headings: {' > '.join(headings)}")
        
        print(f"   ├─ Length: {len(chunk.text)} chars")
        print(f"   └─ Preview: {chunk.text[:150]}...")
    
    # Step 5: Metadata analysis
    print(f"\n5. Metadata Analysis")
    print("-" * 80)
    
    chunks_with_headings = sum(1 for c in chunks if c.meta.get('headings'))
    chunks_with_pages = sum(1 for c in chunks if c.meta.get('page'))
    
    print(f"   Chunks with heading context: {chunks_with_headings} ({chunks_with_headings/len(chunks)*100:.1f}%)")
    print(f"   Chunks with page numbers: {chunks_with_pages} ({chunks_with_pages/len(chunks)*100:.1f}%)")
    
    # Unique pages
    pages = set()
    for chunk in chunks:
        page = chunk.meta.get('page')
        if page:
            pages.add(page)
    print(f"   Unique pages covered: {len(pages)}")
    
    # Step 6: Compare with traditional chunking
    print(f"\n6. Comparison: HybridChunker vs Traditional")
    print("-" * 80)
    
    from services.ingestion.processors.chunker import chunk_text
    
    # Get full text from document
    full_text = docling_doc.export_to_markdown() if hasattr(docling_doc, 'export_to_markdown') else ""
    
    if full_text:
        traditional_chunks = chunk_text(
            full_text,
            chunk_size_tokens=512,
            overlap_tokens=75,
            strategy="token_aware",
        )
        
        print(f"   HybridChunker chunks: {len(chunks)}")
        print(f"   Traditional chunks: {len(traditional_chunks)}")
        print(f"   Difference: {abs(len(chunks) - len(traditional_chunks))}")
        
        # Show metadata richness
        hybrid_meta_keys = set()
        for chunk in chunks:
            hybrid_meta_keys.update(chunk.meta.keys())
        
        trad_meta_keys = set()
        for chunk in traditional_chunks:
            trad_meta_keys.update(chunk.metadata.keys())
        
        print(f"\n   Metadata fields:")
        print(f"   ├─ HybridChunker: {len(hybrid_meta_keys)} fields")
        print(f"   │  {sorted(hybrid_meta_keys)}")
        print(f"   └─ Traditional: {len(trad_meta_keys)} fields")
        print(f"      {sorted(trad_meta_keys)}")
    else:
        print("   (Could not export document to text for comparison)")
    
    print("\n" + "=" * 80)
    print("Test completed successfully! ✓")
    print("=" * 80)


def test_lazy_iteration(pdf_path: str = "tests/pdf/Knoleges.pdf"):
    """Test lazy iteration (memory efficient)."""
    
    print("\n" + "=" * 80)
    print("Testing Lazy Iteration")
    print("=" * 80)
    
    parser = DoclingParser()
    docling_doc = parser.parse_to_document(pdf_path)
    
    chunker = DoclingChunker(max_tokens=512)
    
    print("\nProcessing chunks one by one (lazy)...")
    chunk_count = 0
    for chunk in chunker.chunk_document_lazy(docling_doc):
        chunk_count += 1
        if chunk_count <= 3:
            print(f"  Chunk {chunk_count}: {len(chunk.text)} chars, page {chunk.meta.get('page', 'N/A')}")
    
    print(f"\nProcessed {chunk_count} chunks (lazy iteration)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Docling HybridChunker")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="tests/pdf/Knoleges.pdf",
        help="Path to PDF file to test (default: tests/pdf/Knoleges.pdf)",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Test lazy iteration instead",
    )
    
    args = parser.parse_args()
    
    try:
        if args.lazy:
            test_lazy_iteration(args.pdf_path)
        else:
            test_docling_chunking(args.pdf_path)
    except FileNotFoundError:
        print(f"\n✗ Error: File not found: {args.pdf_path}")
        print("\nAvailable test files:")
        for pdf in Path("tests/pdf").glob("*.pdf"):
            print(f"  - {pdf}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

