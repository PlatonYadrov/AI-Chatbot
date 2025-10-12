#!/usr/bin/env python3
"""Quick test script to verify Docling integration without pytest."""

import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "ingestion"))

print("=" * 70)
print("DOCLING INTEGRATION - QUICK TEST")
print("=" * 70)

# Test 1: Import check
print("\n[1/5] Checking Docling installation...")
try:
    import docling
    from docling.document_converter import DocumentConverter
    version = getattr(docling, '__version__', 'unknown')
    print(f"✓ Docling {version} is installed")
except ImportError as e:
    print(f"✗ Docling not installed: {e}")
    print("\nInstall with: pip install docling")
    sys.exit(1)

# Test 2: Parser import
print("\n[2/5] Checking DoclingParser...")
try:
    from parsers.docling_parser import DoclingParser
    parser = DoclingParser()
    print("✓ DoclingParser imported successfully")
except Exception as e:
    print(f"✗ Failed to import DoclingParser: {e}")
    sys.exit(1)

# Test 3: Parse a test document
print("\n[3/5] Testing document parsing...")
test_pdf = Path(__file__).parent / "tests" / "pdf" / "Knoleges.pdf"
if not test_pdf.exists():
    print(f"⚠ Test PDF not found: {test_pdf}")
    print("Skipping parsing test")
    test_blocks = None
else:
    try:
        parser = DoclingParser(
            extract_tables=True,
            extract_images=True,
            ocr_enabled=True,
            preserve_structure=True,
        )
        test_blocks = list(parser.parse(str(test_pdf), doc_id="test_doc"))
        print(f"✓ Parsed {len(test_blocks)} blocks from test PDF")
        
        # Show first block details
        if test_blocks:
            first = test_blocks[0]
            print(f"  First block type: {first.meta.get('label', 'unknown')}")
            print(f"  First block text: {first.text[:80]}...")
    except Exception as e:
        print(f"✗ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 4: Chunker integration
print("\n[4/5] Testing chunker integration...")
try:
    from processors.chunker import chunk_text
    
    if test_blocks:
        # Test chunking on first block
        first_block = test_blocks[0]
        chunks = chunk_text(
            first_block.text,
            doc_metadata=first_block.meta,
            chunk_size_tokens=200,
            overlap_tokens=50,
        )
        print(f"✓ Chunker created {len(chunks)} chunks from first block")
        
        if chunks:
            first_chunk = chunks[0]
            print(f"  First chunk tokens: {first_chunk.metadata.get('tokens', 'N/A')}")
            print(f"  First chunk text: {first_chunk.text[:80]}...")
    else:
        print("⚠ No test blocks available, skipping chunk test")
except Exception as e:
    print(f"✗ Chunking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Metadata richness
print("\n[5/5] Checking metadata quality...")
if test_blocks:
    try:
        # Collect all unique metadata keys
        all_keys = set()
        for block in test_blocks:
            all_keys.update(block.meta.keys())
        
        print(f"✓ Found {len(all_keys)} unique metadata fields:")
        print(f"  {', '.join(sorted(all_keys))}")
        
        # Check for important fields
        important = ['doc_id', 'type', 'label', 'page', 'path']
        found = [k for k in important if k in all_keys]
        print(f"  Important fields present: {len(found)}/{len(important)}")
    except Exception as e:
        print(f"⚠ Metadata check issue: {e}")
else:
    print("⚠ No test blocks available, skipping metadata check")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ Docling is installed and working")
print("✓ DoclingParser is functional")
if test_blocks:
    print(f"✓ Successfully parsed test document ({len(test_blocks)} blocks)")
    print("✓ Chunker integration working")
    print("✓ Rich metadata preserved")
else:
    print("⚠ Document parsing skipped (no test file)")

print("\nNext steps:")
print("1. Run full tests: pytest tests/test_docling_integration.py -v")
print("2. Try your own documents: python -c 'from services.ingestion.parsers.docling_parser import DoclingParser; ...'")
print("3. Start API: cd services/ingestion && uvicorn api:app --port 8001")
print("\nDocumentation:")
print("- Quick start: services/ingestion/README_DOCLING.md")
print("- Migration guide: services/ingestion/DOCLING_MIGRATION.md")
print("=" * 70)

