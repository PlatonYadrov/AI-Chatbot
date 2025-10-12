"""Minimal test to verify Docling installation and basic functionality."""

from __future__ import annotations

import sys
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "ingestion"))


def test_docling_import():
    """Test that Docling can be imported."""
    try:
        import docling
        from docling.document_converter import DocumentConverter
        print(f"✓ Docling imported successfully (version: {getattr(docling, '__version__', 'unknown')})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Docling: {e}")
        print("Install with: pip install docling")
        return False


def test_docling_parser_import():
    """Test that DoclingParser can be imported."""
    try:
        from parsers.docling_parser import DoclingParser
        parser = DoclingParser()
        print(f"✓ DoclingParser imported and instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import DoclingParser: {e}")
        return False


def test_simple_conversion():
    """Test simple document conversion."""
    try:
        from docling.document_converter import DocumentConverter
        
        # Find a test PDF
        pdf_path = Path(__file__).parent / "pdf" / "Knoleges.pdf"
        if not pdf_path.exists():
            print(f"⚠ Test PDF not found: {pdf_path}")
            return None
        
        print(f"Converting: {pdf_path}")
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document
        
        # Count items
        item_count = sum(1 for _ in doc.iterate_items())
        print(f"✓ Successfully converted document with {item_count} items")
        
        # Show first few items
        print("\nFirst 3 items:")
        for i, (item, level) in enumerate(doc.iterate_items()):
            if i >= 3:
                break
            item_dict = item.model_dump() if hasattr(item, "model_dump") else {}
            label = item_dict.get("label", "unknown")
            text = item_dict.get("text", "")[:50]
            print(f"  {i+1}. [{label}] {text}...")
        
        return True
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Docling Installation & Integration Test")
    print("=" * 60)
    
    print("\n1. Testing Docling import...")
    step1 = test_docling_import()
    
    if not step1:
        print("\n❌ Docling is not installed. Please install it:")
        print("   pip install docling")
        sys.exit(1)
    
    print("\n2. Testing DoclingParser import...")
    step2 = test_docling_parser_import()
    
    print("\n3. Testing simple document conversion...")
    step3 = test_simple_conversion()
    
    print("\n" + "=" * 60)
    if step1 and step2 and step3:
        print("✓ All checks passed! Docling is ready to use.")
    elif step3 is None:
        print("⚠ Docling is installed but test document not found.")
    else:
        print("✗ Some checks failed. See errors above.")
    print("=" * 60)

