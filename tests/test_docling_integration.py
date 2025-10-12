"""Test Docling parser integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "ingestion"))

import pytest


def test_docling_parser_pdf():
    """Test DoclingParser with PDF document."""
    try:
        from parsers.docling_parser import DoclingParser
    except ImportError:
        pytest.skip("Docling not installed")
    
    pdf_path = Path(__file__).parent / "pdf" / "Knoleges.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    parser = DoclingParser(
        extract_tables=True,
        extract_images=True,
        ocr_enabled=True,
        preserve_structure=True,
    )
    
    blocks = list(parser.parse(str(pdf_path), doc_id="test_pdf"))
    
    assert blocks, "Parser should produce at least one block"
    
    for block in blocks[:3]:  # Check first 3 blocks
        assert block.text.strip(), "Block text should not be empty"
        assert block.meta.get("doc_id") == "test_pdf"
        assert block.meta.get("type", "").startswith("docling_")
        assert block.meta.get("path") == str(pdf_path)
        print(f"Block type: {block.meta.get('label')}, text length: {len(block.text)}")


def test_docling_parser_docx():
    """Test DoclingParser with DOCX document."""
    try:
        from parsers.docling_parser import DoclingParser
    except ImportError:
        pytest.skip("Docling not installed")
    
    docx_path = Path(__file__).parent / "docx" / "Darckstore.docx"
    if not docx_path.exists():
        pytest.skip(f"Test DOCX not found: {docx_path}")
    
    parser = DoclingParser(
        extract_tables=True,
        extract_images=True,
        ocr_enabled=True,
        preserve_structure=True,
    )
    
    blocks = list(parser.parse(str(docx_path), doc_id="test_docx"))
    
    assert blocks, "Parser should produce at least one block"
    
    for block in blocks[:3]:
        assert block.text.strip(), "Block text should not be empty"
        assert block.meta.get("doc_id") == "test_docx"
        print(f"Block type: {block.meta.get('label')}, text length: {len(block.text)}")


def test_docling_with_chunker():
    """Test DoclingParser integration with chunker."""
    try:
        from parsers.docling_parser import DoclingParser
        from processors.chunker import chunk_text
    except ImportError:
        pytest.skip("Docling or chunker not available")
    
    pdf_path = Path(__file__).parent / "pdf" / "Knoleges.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    parser = DoclingParser()
    blocks = list(parser.parse(str(pdf_path), doc_id="test_chunking"))
    
    assert blocks, "Parser should produce blocks"
    
    total_chunks = 0
    for block in blocks[:5]:  # Test first 5 blocks
        doc_metadata = {
            **block.meta,
            "source_uri": "test.pdf",
        }
        chunks = chunk_text(
            block.text,
            doc_metadata=doc_metadata,
            chunk_size_tokens=200,
            overlap_tokens=50,
        )
        total_chunks += len(chunks)
        
        for chunk in chunks:
            assert chunk.text.strip(), "Chunk text should not be empty"
            assert chunk.metadata.get("doc_id") == "test_chunking"
            assert "chunk_index" in chunk.metadata
            assert "tokens" in chunk.metadata
    
    print(f"Total chunks from first 5 blocks: {total_chunks}")
    assert total_chunks > 0, "Should produce at least one chunk"


def test_docling_metadata_preservation():
    """Test that Docling preserves rich metadata (headings, labels, etc.)."""
    try:
        from parsers.docling_parser import DoclingParser
    except ImportError:
        pytest.skip("Docling not installed")
    
    pdf_path = Path(__file__).parent / "pdf" / "Knoleges.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    
    parser = DoclingParser(preserve_structure=True)
    blocks = list(parser.parse(str(pdf_path), doc_id="test_metadata"))
    
    # Check that we have blocks with different labels
    labels = {block.meta.get("label") for block in blocks}
    print(f"Found labels: {labels}")
    
    # Check for structured metadata
    has_page_info = any("page" in block.meta for block in blocks)
    has_labels = any("label" in block.meta for block in blocks)
    
    assert has_page_info, "Should have page information"
    assert has_labels, "Should have content labels"


if __name__ == "__main__":
    print("Testing Docling integration...")
    
    print("\n1. Testing PDF parsing...")
    try:
        test_docling_parser_pdf()
        print("✓ PDF parsing test passed")
    except Exception as e:
        print(f"✗ PDF parsing test failed: {e}")
    
    print("\n2. Testing DOCX parsing...")
    try:
        test_docling_parser_docx()
        print("✓ DOCX parsing test passed")
    except Exception as e:
        print(f"✗ DOCX parsing test failed: {e}")
    
    print("\n3. Testing chunker integration...")
    try:
        test_docling_with_chunker()
        print("✓ Chunker integration test passed")
    except Exception as e:
        print(f"✗ Chunker integration test failed: {e}")
    
    print("\n4. Testing metadata preservation...")
    try:
        test_docling_metadata_preservation()
        print("✓ Metadata preservation test passed")
    except Exception as e:
        print(f"✗ Metadata preservation test failed: {e}")
    
    print("\n✓ All tests completed!")

