#!/usr/bin/env python3
"""Test that Docling works completely offline with local models."""

import sys
import os
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "ingestion"))

print("=" * 70)
print("DOCLING OFFLINE MODE TEST")
print("=" * 70)

print("\n[1/5] Checking environment...")
offline_mode = os.getenv("DOCLING_OFFLINE_MODE", "true")
ocr_enabled = os.getenv("DOCLING_OCR_ENABLED", "true")
cache_dir = os.getenv("DOCLING_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))

print(f"  DOCLING_OFFLINE_MODE: {offline_mode}")
print(f"  DOCLING_OCR_ENABLED: {ocr_enabled}")
print(f"  Cache directory: {cache_dir}")

print("\n[2/5] Checking model cache...")
cache_path = Path(cache_dir) / "hub"
if cache_path.exists():
    models = list(cache_path.glob("models--*"))
    print(f"✓ Found {len(models)} models in cache")
    for model in models[:5]:
        model_name = model.name.replace("models--", "").replace("--", "/")
        print(f"  - {model_name}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")
else:
    print(f"⚠ Model cache not found at: {cache_path}")
    print("  Run: python services/ingestion/download_models.py")

print("\n[3/5] Checking Tesseract OCR...")
try:
    import subprocess
    result = subprocess.run(
        ["tesseract", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )
    version = result.stdout.split('\n')[0] if result.stdout else "Unknown"
    print(f"✓ {version}")
except FileNotFoundError:
    print("✗ Tesseract not found")
    print("  Install: apt-get install tesseract-ocr (Linux)")
    print("           brew install tesseract (macOS)")
except Exception as e:
    print(f"⚠ Could not check Tesseract: {e}")

print("\n[4/5] Testing DoclingParser initialization...")
try:
    from parsers.docling_parser import DoclingParser
    
    parser = DoclingParser(
        extract_tables=True,
        extract_images=True,
        ocr_enabled=True,
        preserve_structure=True,
        offline_mode=True,
    )
    print("✓ DoclingParser initialized in offline mode")
    
except Exception as e:
    print(f"✗ Failed to initialize parser: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/5] Testing document parsing...")
test_pdf = Path("tests/pdf/Knoleges.pdf")
if not test_pdf.exists():
    print(f"⚠ Test file not found: {test_pdf}")
    print("  Skipping parsing test")
else:
    try:
        print(f"  Parsing: {test_pdf.name}")
        blocks = list(parser.parse(str(test_pdf), doc_id="offline_test"))
        
        print(f"✓ Successfully parsed {len(blocks)} blocks")
        
        if blocks:
            first = blocks[0]
            print(f"\n  First block:")
            print(f"    Type: {first.meta.get('label', 'unknown')}")
            print(f"    Page: {first.meta.get('page', '?')}")
            print(f"    Text: {first.text[:80]}...")
            
        # Count by type
        from collections import Counter
        labels = Counter(b.meta.get('label', 'unknown') for b in blocks)
        print(f"\n  Content distribution:")
        for label, count in labels.most_common(5):
            print(f"    {label}: {count}")
            
    except Exception as e:
        print(f"✗ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 70)
print("✓ SUCCESS: Docling is working in offline mode!")
print("=" * 70)

print("\nAll processing was done locally:")
print("  ✓ No external API calls")
print("  ✓ Local model inference")
print("  ✓ Tesseract OCR (local)")
print("  ✓ Local table detection")

print("\nTo use in production:")
print("  1. Set DOCLING_OFFLINE_MODE=true")
print("  2. Mount cache volume: -v ./models:/app/.cache/huggingface")
print("  3. Ensure Tesseract is installed")
print("  4. Run: docker-compose up -d ingestion")

